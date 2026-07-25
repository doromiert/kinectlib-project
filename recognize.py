"""
recognize.py — face recognition against enrolled embeddings

loads all .npz files from enroll/<name>/ and matches live insightface
embeddings via cosine similarity. always returns best match, flags low
confidence instead of dropping.

usage:
    from recognize import Recognizer

    rec = Recognizer()  # loads from ./enroll/ by default
    rec = Recognizer("path/to/enroll/dir")

    with Kinect() as k, PoseTracker() as pose_tracker:
        rgb, depth, ir = k.get_frames()
        results = rec.process(rgb, depth, ir)

        for r in results:
            print(r["name"])        # str, e.g. "Alice" or "unknown"
            print(r["confidence"])  # float 0.0-1.0 (cosine similarity)
            print(r["known"])       # False if below threshold
            print(r["bbox"])        # (x1,y1,x2,y2) in RGB px
            print(r["embedding"])   # np.ndarray (512,)
            print(r["body"])        # body dict from PoseTracker, or None
"""

import os
import glob
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

import sys
sys.path.insert(0, ".")
from kinect import Kinect
from pose import PoseTracker, associate_faces_to_bodies
from hands import _is_dark, _ir_to_detection_image

# cosine similarity below this = "known=False" but still returned
CONFIDENCE_THRESHOLD = 0.45

# insightface det_size — 640 is the sweet spot, drop to 320 if too slow
DET_SIZE = (640, 640)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


class Recognizer:
    def __init__(self, enroll_dir: str = "enroll", providers=None):
        """
        enroll_dir: root dir containing per-person subdirs with .npz files
                    e.g. enroll/Alice/Alice_c0r0_00.npz
        providers:  onnxruntime providers list, defaults to CPU
                    pass ["ROCMExecutionProvider", "CPUExecutionProvider"] for AMD GPU
        """
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self._face_app = FaceAnalysis(name="buffalo_l", providers=providers)
        self._face_app.prepare(ctx_id=0, det_size=DET_SIZE)

        self._gallery = self._load_gallery(enroll_dir)
        names = list(self._gallery.keys())
        total = sum(len(v) for v in self._gallery.values())
        print(f"recognizer loaded: {len(names)} people, {total} embeddings")
        print(f"  people: {', '.join(names) if names else '(none)'}")

    # ── gallery ───────────────────────────────────────────────────────────────

    def _load_gallery(self, enroll_dir: str) -> dict[str, np.ndarray]:
        """
        returns {name: (N, 512) mean-pooled embedding matrix}
        mean-pools all captures per cell so each cell contributes one vector,
        then stacks them — so the gallery isn't biased toward cells with more captures.
        """
        gallery = {}

        if not os.path.isdir(enroll_dir):
            print(f"warning: enroll_dir '{enroll_dir}' not found — gallery empty")
            return gallery

        for person_dir in sorted(os.scandir(enroll_dir), key=lambda e: e.name):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            npz_files = glob.glob(os.path.join(person_dir.path, "*.npz"))
            if not npz_files:
                continue

            # group by cell
            cell_embeddings: dict[tuple, list] = {}
            for path in npz_files:
                data = np.load(path)
                emb  = data["embedding"]           # (512,)
                cell = tuple(data["cell"].tolist()) # (col, row)
                cell_embeddings.setdefault(cell, []).append(emb)

            # mean-pool per cell, stack
            pooled = [np.mean(embs, axis=0) for embs in cell_embeddings.values()]
            gallery[name] = np.stack(pooled)  # (n_cells, 512)

        return gallery

    def reload(self, enroll_dir: str = "enroll"):
        """hot-reload gallery without restarting (call after new enrollment)"""
        self._gallery = self._load_gallery(enroll_dir)

    # ── matching ──────────────────────────────────────────────────────────────

    def _match(self, embedding: np.ndarray) -> tuple[str, float]:
        """
        returns (name, confidence) — always returns best match even if unknown.
        confidence = max cosine similarity across all gallery embeddings for that person
        (using max not mean because pose/lighting variation means one cell will always
        match better than average)
        """
        best_name  = "unknown"
        best_score = -1.0

        for name, gallery_embs in self._gallery.items():
            # score against each cell embedding, take max
            scores = [_cosine(embedding, g) for g in gallery_embs]
            score  = max(scores)
            if score > best_score:
                best_score = score
                best_name  = name

        return best_name, best_score

    # ── main API ──────────────────────────────────────────────────────────────

    def process(self, rgb_frame: np.ndarray, depth_frame: np.ndarray,
                ir_frame: np.ndarray = None) -> list[dict]:
        """
        rgb_frame:   (1080, 1920, 4) uint8  BGRX
        depth_frame: (424, 512)      float32 mm
        ir_frame:    (424, 512)      float32 — optional

        returns list of result dicts, one per detected face:
        {
            "name":       str,           # best match name, or "unknown"
            "confidence": float,         # cosine similarity 0.0-1.0
            "known":      bool,          # False if below CONFIDENCE_THRESHOLD
            "bbox":       (x1,y1,x2,y2),
            "embedding":  np.ndarray,    # raw live embedding
            "body":       dict | None,   # PoseTracker body dict
        }
        """
        self._ensure_pose_tracker()
        bgr = rgb_frame[:, :, :3]

        use_ir = ir_frame is not None and _is_dark(rgb_frame)
        detect_img = _ir_to_detection_image(ir_frame) if use_ir \
                     else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        faces_raw = self._face_app.get(detect_img)
        if not faces_raw:
            return []

        faces = [
            {
                "bbox":      tuple(map(int, f.bbox)),
                "embedding": f.embedding,
                "kps":       f.kps,
            }
            for f in faces_raw
        ]

        bodies      = self._pose_tracker.process(rgb_frame, depth_frame, ir_frame) \
                      if self._pose_tracker else []
        associations = associate_faces_to_bodies(faces, bodies)

        results = []
        for assoc in associations:
            face = assoc["face"]
            body = assoc["body"]
            emb  = face["embedding"]

            if self._gallery:
                name, confidence = self._match(emb)
            else:
                name, confidence = "unknown", 0.0

            results.append({
                "name":       name,
                "confidence": round(confidence, 4),
                "known":      confidence >= CONFIDENCE_THRESHOLD,
                "bbox":       face["bbox"],
                "embedding":  emb,
                "body":       body,
                "source":     "ir" if use_ir else "rgb",
            })

        return results

    # ── context manager — owns pose tracker lifecycle ─────────────────────────

    def __enter__(self):
        self._pose_tracker = PoseTracker()
        return self

    def __exit__(self, *_):
        if self._pose_tracker:
            self._pose_tracker.close()
            self._pose_tracker = None

    # allow use without context manager (no pose tracking)
    # _pose_tracker is always initialized; None = no pose tracking
    def _ensure_pose_tracker(self):
        if not hasattr(self, "_pose_tracker"):
            self._pose_tracker = None


# ── drawing helper ────────────────────────────────────────────────────────────

def draw_results(bgr: np.ndarray, results: list[dict]) -> np.ndarray:
    """
    draws all recognition info onto bgr frame, returns annotated copy.
    per face:
      - bbox rectangle (green=known, blue=unknown)
      - name + confidence score
      - known/unknown badge
      - face bbox size in px
    per body (if present):
      - nose landmark dot
      - shoulder-to-shoulder line
      - wrist dots (left/right)
      - per-landmark Z depth labels at key points
    """
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    COLOR_KNOWN   = (60,  210, 60)   # green
    COLOR_UNKNOWN = (60,  60,  210)  # blue
    COLOR_BODY    = (180, 180, 60)   # amber for skeleton
    COLOR_WRIST   = (60,  180, 210)  # cyan for wrists

    out = bgr.copy()

    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        color = COLOR_KNOWN if r["known"] else COLOR_UNKNOWN
        bw, bh = x2 - x1, y2 - y1

        # face bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

        # name + confidence — above bbox
        src_tag = f" [{r.get('source', 'rgb')}]"
        name_label = f"{'✓' if r['known'] else '?'}  {r['name']}  ({r['confidence']:.3f}){src_tag}"
        cv2.putText(out, name_label, (x1, y1 - 36), FONT, 0.9, color, 2, cv2.LINE_AA)

        # bbox size + known badge — below name
        meta_label = f"{'KNOWN' if r['known'] else 'UNKNOWN'}  bbox:{bw}×{bh}px"
        cv2.putText(out, meta_label, (x1, y1 - 10), FONT, 0.65, color, 1, cv2.LINE_AA)

        body = r["body"]
        if body:
            lms = body["landmarks"]

            # shoulder line
            lsx, lsy, lsz = lms["left_shoulder"]
            rsx, rsy, rsz = lms["right_shoulder"]
            cv2.line(out, (lsx, lsy), (rsx, rsy), COLOR_BODY, 2, cv2.LINE_AA)

            # nose
            nx, ny, nz = body["nose"]
            cv2.circle(out, (nx, ny), 8, COLOR_BODY, -1)
            cv2.putText(out, f"nose {nz:.0f}mm", (nx + 10, ny),
                        FONT, 0.55, COLOR_BODY, 1, cv2.LINE_AA)

            # wrists
            for side, key, wcolor in [
                ("L", "left_wrist",  COLOR_WRIST),
                ("R", "right_wrist", COLOR_WRIST),
            ]:
                wx, wy, wz = lms[key]
                cv2.circle(out, (wx, wy), 7, wcolor, -1)
                cv2.putText(out, f"{side}wrist {wz:.0f}mm", (wx + 10, wy),
                            FONT, 0.55, wcolor, 1, cv2.LINE_AA)

            # shoulder Z labels
            cv2.putText(out, f"Lsh {lsz:.0f}mm", (lsx - 100, lsy - 10),
                        FONT, 0.5, COLOR_BODY, 1, cv2.LINE_AA)
            cv2.putText(out, f"Rsh {rsz:.0f}mm", (rsx + 8,   rsy - 10),
                        FONT, 0.5, COLOR_BODY, 1, cv2.LINE_AA)

            # hips
            lhx, lhy, lhz = lms["left_hip"]
            rhx, rhy, rhz = lms["right_hip"]
            cv2.line(out, (lhx, lhy), (rhx, rhy), COLOR_BODY, 1, cv2.LINE_AA)
            cv2.circle(out, (lhx, lhy), 5, COLOR_BODY, -1)
            cv2.circle(out, (rhx, rhy), 5, COLOR_BODY, -1)

            # torso lines (shoulder→hip)
            cv2.line(out, (lsx, lsy), (lhx, lhy), COLOR_BODY, 1, cv2.LINE_AA)
            cv2.line(out, (rsx, rsy), (rhx, rhy), COLOR_BODY, 1, cv2.LINE_AA)

            # detection source badge bottom-left of face bbox
            src = body.get("source", "rgb")
            cv2.putText(out, f"src:{src}", (x1, y2 + 18),
                        FONT, 0.55, color, 1, cv2.LINE_AA)

    return out


# ── standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time
    sys.path.insert(0, ".")
    from kinect import Kinect

    COUNTDOWN_S = 5

    print("opening kinect...")
    with Kinect() as kinect, Recognizer() as rec:
        print("warming up...")
        for _ in range(30):
            kinect.get_frames()

        for i in range(COUNTDOWN_S, 0, -1):
            print(f"get in position... {i}", end="\r", flush=True)
            time.sleep(1)
        print("\ngrabbing frame...")

        rgb, depth, ir = kinect.get_frames()

        if rgb is None:
            print("timeout")
            sys.exit(1)

        results = rec.process(rgb, depth, ir)

        if not results:
            print("no faces detected")
        else:
            for r in results:
                status = "✓" if r["known"] else "?"
                print(f"\n{status} {r['name']}  ({r['confidence']:.3f})")
                print(f"  known:     {r['known']}")
                print(f"  bbox:      {r['bbox']}")
                if r["body"]:
                    nx, ny, nz = r["body"]["nose"]
                    lx, ly, lz = r["body"]["left_wrist"]
                    rx, ry, rz = r["body"]["right_wrist"]
                    print(f"  nose:      ({nx}, {ny}, {nz:.0f}mm)")
                    print(f"  L wrist:   ({lx}, {ly}, {lz:.0f}mm)")
                    print(f"  R wrist:   ({rx}, {ry}, {rz:.0f}mm)")
                    print(f"  src:       {r['body'].get('source', 'rgb')}")

        out = draw_results(rgb[:, :, :3], results)
        cv2.imwrite("recognize.png", out)
        print("\nsaved recognize.png")
