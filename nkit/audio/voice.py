"""
nkit/audio/voice.py — wake word detection + STT, with self-sound cancellation

Listens on the mic 24/7 with openwakeword (near-zero CPU). On detection,
records until silence (silero VAD), then transcribes with faster-whisper.
Two wake words map to two modes:

  "hey zane"   -> mode="assistant"  (command to Zane)
  "zane write" -> mode="stt"        (dictation)

Delivery is callback-only — nkit never touches the OS (no typing, no
injection). What you do with a "stt" transcript is entirely up to the app.

Self-sound cancellation (AEC) runs ahead of everything else in this
pipeline: if an AecCalibration is supplied, a background thread taps the
system loopback/monitor source and feeds it to nkit.audio.aec.EchoCanceller,
which cleans every mic frame before wake word detection, VAD, or whisper
ever see it. Without a calibration, mic audio passes through unmodified —
AEC is opt-in, not required to use voice at all.

usage:
    from nkit.audio.voice import VoiceListener
    from nkit.audio import aec

    calibration = aec.calibrate()   # run once per audio setup

    def on_result(result):
        print(result.mode, result.text)

    with VoiceListener(
        hey_zane_model="wakeword_models/hey_zane.onnx",
        zane_write_model="wakeword_models/zane_write.onnx",
        callback=on_result,
        aec_calibration=calibration,
    ) as v:
        v.start()
        ...
"""

from __future__ import annotations
import queue
import threading
import time
from typing import Callable

import numpy as np
import pyaudio

from ..types import WakeWordResult, AecCalibration
from . import aec as _aec

_oww     = None
_whisper = None
_silero  = None
_torch   = None


SAMPLE_RATE = 16000
CHANNELS    = 1
# shared with aec.AEC_FRAME_SIZE (32ms) and silero VAD's hard 512-sample
# requirement, so the whole pipeline — AEC, openwakeword, VAD — runs on one
# consistent frame size instead of juggling three. openwakeword's own
# examples typically use 1280-sample (80ms) frames; its predict() is
# documented as tolerant of streaming chunks of other sizes, but this
# specific size hasn't been verified against real hardware — worth
# confirming detection quality once you can test on the actual mic.
CHUNK = _aec.AEC_FRAME_SIZE

VAD_THRESHOLD      = 0.5
VAD_SILENCE_S      = 1.2
VAD_MAX_DURATION_S = 15.0

OWW_THRESHOLD = 0.5

WHISPER_MODEL    = "base"
WHISPER_LANGUAGE = None
WHISPER_DEVICE   = "cpu"
WHISPER_COMPUTE  = "int8"


def _lazy_imports():
    global _oww, _whisper, _silero, _torch
    if _oww is None:
        import openwakeword
        _oww = openwakeword
    if _whisper is None:
        import faster_whisper
        _whisper = faster_whisper
    if _silero is None:
        import torch
        _torch = torch
        silero_model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False,
        )
        _silero = silero_model


def _vad_is_speech(chunk_f32: np.ndarray) -> float:
    t = _torch.from_numpy(chunk_f32)
    with _torch.no_grad():
        return _silero(t, SAMPLE_RATE).item()


def _transcribe(audio_bytes: bytes, whisper_model) -> dict:
    audio_f32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments, info = whisper_model.transcribe(
        audio_f32, language=WHISPER_LANGUAGE, word_timestamps=True, vad_filter=True,
    )
    words = []
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text.strip())
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word, "start": round(w.start, 3),
                    "end": round(w.end, 3), "probability": round(w.probability, 4),
                })
    return {"text": " ".join(text_parts).strip(), "language": info.language, "words": words}


class VoiceListener:
    def __init__(
        self,
        hey_zane_model:   str | None,
        zane_write_model: str | None,
        callback:            Callable[[WakeWordResult], None] | None = None,
        raw_audio_callback:  Callable[[bytes], None] | None = None,
        aec_calibration:     AecCalibration | None = None,
        whisper_model:    str   = WHISPER_MODEL,
        whisper_device:   str   = WHISPER_DEVICE,
        whisper_compute:  str   = WHISPER_COMPUTE,
        oww_threshold:    float = OWW_THRESHOLD,
        vad_threshold:    float = VAD_THRESHOLD,
        vad_silence_s:    float = VAD_SILENCE_S,
        mic_index:        int   = -1,
        monitor_index:    int   = -1,
    ):
        """
        hey_zane_model / zane_write_model: path to .onnx wake word models,
            or both None for no-model mode — mic still opens and gets
            cleaned by AEC if configured, only raw_audio_callback fires.
        aec_calibration: result of aec.calibrate(). None disables AEC.
        mic_index / monitor_index: -1 = autodetect (kinect mic by name,
            monitor by "*.monitor" name).
        """
        self._model_paths = {
            k: v for k, v in {"hey_zane": hey_zane_model, "zane_write": zane_write_model}.items() if v
        }
        self._callback           = callback
        self._raw_audio_callback = raw_audio_callback
        self._aec_calibration    = aec_calibration
        self._oww_threshold      = oww_threshold
        self._vad_threshold      = vad_threshold
        self._vad_silence_s      = vad_silence_s
        self._mic_index          = mic_index
        self._monitor_index      = monitor_index
        self._whisper_cfg        = (whisper_model, whisper_device, whisper_compute)

        self._stop_event  = threading.Event()
        self._work_queue  = queue.Queue()
        self._listen_thread  = None
        self._monitor_thread = None
        self._process_thread = None
        self._aec = None
        self._pa  = None

    def start(self):
        """start background threads — non-blocking"""
        _lazy_imports()

        model_size, device, compute = self._whisper_cfg
        self._whisper_model = _whisper.WhisperModel(model_size, device=device, compute_type=compute)

        if self._model_paths:
            self._oww_model = _oww.Model(
                wakeword_models=list(self._model_paths.values()), inference_framework="onnx",
            )
            self._ww_names = list(self._model_paths.keys())
        else:
            self._oww_model = None
            self._ww_names = []

        self._pa = pyaudio.PyAudio()

        if self._aec_calibration is not None:
            self._aec = _aec.EchoCanceller(self._aec_calibration, frame_size=CHUNK, sample_rate=SAMPLE_RATE)

        self._stop_event.clear()
        self._listen_thread  = threading.Thread(target=self._listen_loop,  daemon=True)
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._listen_thread.start()
        self._process_thread.start()

        if self._aec is not None:
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()

        print("voice listener started" + ("  (AEC on)" if self._aec else "  (AEC off)"))

    def stop(self):
        self._stop_event.set()
        if self._listen_thread:
            self._listen_thread.join(timeout=2.0)
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        if self._process_thread:
            self._work_queue.put(None)   # poison pill
            self._process_thread.join(timeout=5.0)
        if self._aec:
            self._aec.close()
            self._aec = None
        if self._pa:
            self._pa.terminate()
            self._pa = None
        print("voice listener stopped")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()

    # ── loopback monitor capture, feeds the AEC reference buffer ───────────────

    def _monitor_loop(self):
        monitor_idx = self._monitor_index if self._monitor_index >= 0 else _aec.find_monitor_source(self._pa)
        if monitor_idx is None:
            print("[voice] no monitor source found — disabling AEC")
            self._aec = None
            return

        stream = self._pa.open(
            format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLE_RATE, input=True,
            input_device_index=monitor_idx, frames_per_buffer=CHUNK,
        )
        print(f"listening on monitor [{monitor_idx}]  (AEC reference)")
        try:
            while not self._stop_event.is_set():
                raw = stream.read(CHUNK, exception_on_overflow=False)
                if self._aec:
                    self._aec.push_reference(raw)
        finally:
            stream.stop_stream()
            stream.close()

    # ── listen loop (always on, openwakeword) ───────────────────────────────────

    def _listen_loop(self):
        mic = self._mic_index if self._mic_index >= 0 else _aec.find_kinect_mic(self._pa)
        stream = self._pa.open(
            format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLE_RATE, input=True,
            input_device_index=mic, frames_per_buffer=CHUNK,
        )
        print(f"listening on mic [{mic}]" + ("  (openwakeword)" if self._oww_model else "  (raw audio only)"))

        try:
            while not self._stop_event.is_set():
                raw = stream.read(CHUNK, exception_on_overflow=False)
                if self._aec:
                    raw = self._aec.process(raw)

                if self._raw_audio_callback:
                    self._raw_audio_callback(raw)

                if self._oww_model is None:
                    continue

                pcm = np.frombuffer(raw, dtype=np.int16)
                predictions = self._oww_model.predict(pcm)

                for ww_name in self._ww_names:
                    score = predictions.get(ww_name, 0.0)
                    if score >= self._oww_threshold:
                        print(f"[oww] '{ww_name}' detected  score={score:.3f}")
                        self._oww_model.reset()
                        audio = self._capture_command(stream)
                        self._work_queue.put((ww_name, score, audio))
                        break   # one detection at a time
        finally:
            stream.stop_stream()
            stream.close()

    # ── capture until silence (silero VAD) ──────────────────────────────────────

    def _capture_command(self, stream) -> bytes:
        """record until VAD silence or max duration, return raw int16 bytes (AEC-cleaned if enabled)"""
        frames        = []
        silence_start = None
        t_start       = time.time()

        while True:
            raw = stream.read(CHUNK, exception_on_overflow=False)
            if self._aec:
                raw = self._aec.process(raw)
            if self._raw_audio_callback:
                self._raw_audio_callback(raw)
            frames.append(raw)

            pcm_f32 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            prob    = _vad_is_speech(pcm_f32)
            now = time.time()

            if prob >= self._vad_threshold:
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = now
                elif now - silence_start >= self._vad_silence_s:
                    break

            if now - t_start >= VAD_MAX_DURATION_S:
                break

        return b"".join(frames)

    # ── process loop (transcription, off main thread) ─────────────────────────────

    def _process_loop(self):
        while True:
            item = self._work_queue.get()
            if item is None:
                break
            ww_name, score, audio_bytes = item
            mode = "assistant" if ww_name == "hey_zane" else "stt"

            try:
                result = _transcribe(audio_bytes, self._whisper_model)
            except Exception as e:
                print(f"[voice] transcription error: {e}")
                continue

            event = WakeWordResult(
                mode=mode, wake_word=ww_name, text=result["text"], language=result["language"],
                words=result["words"], confidence=round(score, 4),
            )

            if self._callback:
                try:
                    self._callback(event)
                except Exception as e:
                    print(f"[voice] callback error: {e}")
