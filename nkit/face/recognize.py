"""
nkit/face/recognize.py — face recognition against enrolled embeddings

Loads all .npz files from enroll_dir/<name>/ and matches live insightface
embeddings via cosine similarity. Always returns the best match, flags low
confidence via known=False. enroll_dir is fully caller-specifiable — this
is what makes "dir of known hashes" configurable per app.

usage:
    from nkit.face.recognize import Recognizer

    with Recognizer(enroll_dir="enroll") as rec:
        faces = rec.process(rgb, depth, ir, config, bodies)
        for f in faces:
            print(f.name, f.confidence, f.known, f.skeleton_id)
"""

from __future__ import annotations
import os
import glob
from typing import Literal

import numpy as np
from insightface.app import FaceAnalysis

from ..types import FaceResult, BodyResult, GestureConfig
from .. import _vision
from .._vision import Roi
from ..body import associate_faces_to_bodies

CONFIDENCE_THRESHOLD = 0.45
DET_SIZE = (640, 640)

# half-width (px) of the crop searched around a skeleton's own nose
# estimate when bodies are supplied to process() — same idea as
# hands.py's WRIST_CROP_HALF_PX: a face-like false positive elsewhere in
# frame is never in the search region at all.
HEAD_CROP_HALF_PX = 260


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


class Recognizer:
    def __init__(
        self,
        enroll_dir: str  = "enroll",
        providers:  list | None = None,
    ):
        """
        enroll_dir: root dir containing per-person subdirs with .npz files —
                    fully caller-specifiable, this is the "known hashes" gallery.
        providers:  onnxruntime providers list, defaults to CPU.
                    for AMD GPU: ["ROCMExecutionProvider", "CPUExecutionProvider"]
        """
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self._face_app = FaceAnalysis(name="buffalo_l", providers=providers)
        self._face_app.prepare(ctx_id=0, det_size=DET_SIZE)
        self._roi_tracker = _vision.PersonRoiTracker()

        self._enroll_dir = enroll_dir
        self._gallery    = self._load_gallery(enroll_dir)

        names = list(self._gallery.keys())
        total = sum(len(v) for v in self._gallery.values())
        print(f"recognizer: {len(names)} people, {total} embeddings")
        if names:
            print(f"  people: {', '.join(names)}")

    # ── gallery ───────────────────────────────────────────────────────────────

    def _load_gallery(self, enroll_dir: str) -> dict[str, np.ndarray]:
        """
        returns {name: (N, 512) stacked embedding matrix}
        mean-pools per grid cell so each cell contributes one vector.
        """
        gallery: dict[str, np.ndarray] = {}

        if not os.path.isdir(enroll_dir):
            print(f"warning: enroll_dir '{enroll_dir}' not found — gallery empty")
            return gallery

        for entry in sorted(os.scandir(enroll_dir), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            name      = entry.name
            npz_files = glob.glob(os.path.join(entry.path, "*.npz"))
            if not npz_files:
                continue

            cell_embeddings: dict[tuple, list] = {}
            for path in npz_files:
                data = np.load(path)
                emb  = data["embedding"]
                cell = tuple(data["cell"].tolist())
                cell_embeddings.setdefault(cell, []).append(emb)

            pooled = [np.mean(embs, axis=0) for embs in cell_embeddings.values()]
            gallery[name] = np.stack(pooled)

        return gallery

    def reload(self):
        """hot-reload gallery without restarting (call after new enrollment)."""
        self._gallery = self._load_gallery(self._enroll_dir)

    # ── matching ──────────────────────────────────────────────────────────────

    def _match(self, embedding: np.ndarray) -> tuple[str, float]:
        best_name  = "unknown"
        best_score = -1.0

        for name, gallery_embs in self._gallery.items():
            scores = [_cosine(embedding, g) for g in gallery_embs]
            score  = max(scores)
            if score > best_score:
                best_score = score
                best_name  = name

        return best_name, best_score

    # ── main API ──────────────────────────────────────────────────────────────

    def process(
        self,
        rgb_frame:   np.ndarray,
        depth_frame: np.ndarray,
        ir_frame:    np.ndarray | None = None,
        config:      GestureConfig | None = None,
        bodies:      list[BodyResult] | None = None,
        roi:         Roi | None | Literal["auto"] = "auto",
    ) -> list[FaceResult]:
        """
        rgb_frame:   (1080, 1920, 4) uint8  BGRX
        depth_frame: (424, 512)      float32 mm
        ir_frame:    (424, 512)      float32  — optional
        config:      GestureConfig — thresholds; defaults if omitted
        bodies:      list of BodyResult from BodyTracker. When given, face
                     detection runs in a small crop around EACH body's own
                     nose estimate instead of one big region — same idea as
                     hands.py's wrist cropping, rejects a face-like false
                     positive elsewhere in frame by never looking there —
                     and results get associated back to those bodies. Pass
                     None or [] to search one region (roi below) and skip
                     skeleton association (nkit/bridge.py does this when
                     skeleton tracking is off).
        roi: only used when bodies is empty. "auto" (default) crops to a
             depth-based person region before detection; pass an explicit
             (x1,y1,x2,y2) to reuse one shared with body/hand tracking this
             frame, or None to disable cropping.

        returns list of FaceResult, one per detected face.
        """
        config = config or GestureConfig()

        if bodies:
            faces: list[dict] = []
            for body in bodies:
                crop = _vision.point_crop_roi(body.nose.x, body.nose.y, HEAD_CROP_HALF_PX)
                faces.extend(self._detect_in_roi(rgb_frame, ir_frame, config, crop))
        else:
            resolved_roi = self._roi_tracker.find(depth_frame) if roi == "auto" else roi
            faces = self._detect_in_roi(rgb_frame, ir_frame, config, resolved_roi)

        if not faces:
            return []

        associations = associate_faces_to_bodies(faces, bodies or [])

        results = []
        for assoc in associations:
            face = assoc["face"]
            body = assoc["body"]
            emb  = face["embedding"]

            if self._gallery:
                name, confidence = self._match(emb)
            else:
                name, confidence = "unknown", 0.0

            results.append(FaceResult(
                name        = name,
                confidence  = round(confidence, 4),
                known       = confidence >= CONFIDENCE_THRESHOLD,
                bbox        = face["bbox"],
                embedding   = emb,
                source      = face["source"],
                skeleton_id = body.skeleton_id if body else None,
            ))

        return results

    def _detect_in_roi(self, rgb_frame: np.ndarray, ir_frame: np.ndarray | None, config: GestureConfig, roi: Roi | None) -> list[dict]:
        detect_img, source, roi_used = _vision.detection_image(rgb_frame, ir_frame, config, roi)
        ox, oy = roi_used[0], roi_used[1]   # insightface returns pixel coords relative to detect_img, offset back to full RGB space

        faces_raw = self._face_app.get(detect_img)
        if not faces_raw:
            return []

        return [
            {
                "bbox":      (int(f.bbox[0]) + ox, int(f.bbox[1]) + oy, int(f.bbox[2]) + ox, int(f.bbox[3]) + oy),
                "embedding": f.embedding,
                "kps":       f.kps + [ox, oy],
                "source":    source,
            }
            for f in faces_raw
        ]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
