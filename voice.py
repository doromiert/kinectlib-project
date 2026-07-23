"""
voice.py — wake word detection + STT pipeline

listens on kinect mic 24/7 with openwakeword (near-zero CPU).
on detection, records until silence (silero VAD), then transcribes
with faster-whisper.

two modes triggered by different wake words:
  "hey zane"   → mode="assistant"  (command to Zane assistant)
  "zane write" → mode="stt"        (dictation / text input)

usage:
    from voice import VoiceListener

    def on_result(result):
        print(result["mode"])        # "assistant" | "stt"
        print(result["text"])        # transcript
        print(result["language"])    # "en" | "pl" | etc.
        print(result["words"])       # [{word, start, end, probability}, ...]
        print(result["wake_word"])   # "hey_zane" | "zane_write"
        print(result["confidence"])  # openwakeword score that triggered

    listener = VoiceListener(
        hey_zane_model="wakeword_models/hey_zane.onnx",
        zane_write_model="wakeword_models/zane_write.onnx",
        callback=on_result,
    )
    listener.start()   # non-blocking, runs in background thread
    ...
    listener.stop()

    # or as context manager:
    with VoiceListener(..., callback=on_result) as v:
        v.start()
        time.sleep(60)
"""

import os
import io
import queue
import threading
import time
import wave
import numpy as np
import pyaudio

# lazy imports — pulled in at init time so missing deps fail clearly
_oww       = None
_whisper   = None
_silero    = None
_torch     = None


SAMPLE_RATE   = 16000
CHANNELS      = 1
CHUNK         = 1280   # openwakeword wants 80ms frames at 16kHz = 1280 samples

# VAD config
VAD_THRESHOLD      = 0.5    # silero confidence threshold
VAD_SILENCE_S      = 1.2    # seconds of silence before stopping capture
VAD_MAX_DURATION_S = 15.0   # hard cap

# openwakeword
OWW_THRESHOLD = 0.5   # detection confidence threshold

# whisper
WHISPER_MODEL    = "base"   # tiny/base/small — base is the sweet spot for target hw
WHISPER_LANGUAGE = None     # None = auto-detect (handles EN+PL fine)
WHISPER_DEVICE   = "cpu"    # "cuda" / "rocm" if available — set via env
WHISPER_COMPUTE  = "int8"   # int8 = fast on CPU


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
        silero_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        _silero = silero_model


def _find_kinect_mic(pa: pyaudio.PyAudio) -> int:
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        name = info["name"].lower()
        if "kinect" in name or "xbox" in name or "nui" in name:
            return i
    return pa.get_default_input_device_info()["index"]


def _vad_is_speech(chunk_f32: np.ndarray) -> float:
    """returns silero speech probability for a 16kHz chunk"""
    t = _torch.from_numpy(chunk_f32)
    with _torch.no_grad():
        prob = _silero(t, SAMPLE_RATE).item()
    return prob


def _transcribe(audio_bytes: bytes, whisper_model) -> dict:
    """transcribe raw 16kHz int16 bytes, return {text, language, words}"""
    audio_f32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    segments, info = whisper_model.transcribe(
        audio_f32,
        language=WHISPER_LANGUAGE,
        word_timestamps=True,
        vad_filter=True,
    )

    words = []
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text.strip())
        if seg.words:
            for w in seg.words:
                words.append({
                    "word":        w.word,
                    "start":       round(w.start, 3),
                    "end":         round(w.end, 3),
                    "probability": round(w.probability, 4),
                })

    return {
        "text":     " ".join(text_parts).strip(),
        "language": info.language,
        "words":    words,
    }


class VoiceListener:
    def __init__(
        self,
        hey_zane_model:   str,
        zane_write_model: str,
        callback,
        whisper_model:    str  = WHISPER_MODEL,
        whisper_device:   str  = WHISPER_DEVICE,
        whisper_compute:  str  = WHISPER_COMPUTE,
        oww_threshold:    float = OWW_THRESHOLD,
        vad_threshold:    float = VAD_THRESHOLD,
        vad_silence_s:    float = VAD_SILENCE_S,
        mic_index:        int   = -1,
    ):
        """
        hey_zane_model:   path to hey_zane.onnx
        zane_write_model: path to zane_write.onnx
        callback:         fn(result_dict) called in a worker thread on each detection
        mic_index:        -1 = autodetect kinect mic
        """
        self._model_paths = {
            "hey_zane":   hey_zane_model,
            "zane_write": zane_write_model,
        }
        self._callback      = callback
        self._oww_threshold = oww_threshold
        self._vad_threshold = vad_threshold
        self._vad_silence_s = vad_silence_s
        self._mic_index     = mic_index
        self._whisper_cfg   = (whisper_model, whisper_device, whisper_compute)

        self._stop_event  = threading.Event()
        self._work_queue  = queue.Queue()
        self._listen_thread  = None
        self._process_thread = None

    def start(self):
        """start background threads — non-blocking"""
        _lazy_imports()

        # load whisper once
        model_size, device, compute = self._whisper_cfg
        self._whisper_model = _whisper.WhisperModel(
            model_size, device=device, compute_type=compute
        )

        # load openwakeword
        self._oww_model = _oww.Model(
            wakeword_models=list(self._model_paths.values()),
            inference_framework="onnx",
        )
        self._ww_names = list(self._model_paths.keys())

        self._stop_event.clear()
        self._listen_thread  = threading.Thread(target=self._listen_loop,   daemon=True)
        self._process_thread = threading.Thread(target=self._process_loop,  daemon=True)
        self._listen_thread.start()
        self._process_thread.start()
        print("voice listener started")

    def stop(self):
        self._stop_event.set()
        if self._listen_thread:
            self._listen_thread.join(timeout=2.0)
        if self._process_thread:
            self._work_queue.put(None)   # poison pill
            self._process_thread.join(timeout=5.0)
        print("voice listener stopped")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()

    # ── listen loop (always on, openwakeword) ─────────────────────────────────

    def _listen_loop(self):
        pa = pyaudio.PyAudio()
        mic = self._mic_index if self._mic_index >= 0 else _find_kinect_mic(pa)
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=mic,
            frames_per_buffer=CHUNK,
        )
        print(f"listening on mic [{mic}]  (openwakeword)")

        try:
            while not self._stop_event.is_set():
                raw = stream.read(CHUNK, exception_on_overflow=False)
                pcm = np.frombuffer(raw, dtype=np.int16)

                predictions = self._oww_model.predict(pcm)

                for ww_name in self._ww_names:
                    score = predictions.get(ww_name, 0.0)
                    if score >= self._oww_threshold:
                        print(f"[oww] '{ww_name}' detected  score={score:.3f}")
                        self._oww_model.reset()   # clear buffer after detection
                        audio = self._capture_command(stream)
                        self._work_queue.put((ww_name, score, audio))
                        break   # one detection at a time
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    # ── capture until silence (silero VAD) ───────────────────────────────────

    def _capture_command(self, stream) -> bytes:
        """record until VAD silence or max duration, return raw int16 bytes"""
        frames        = []
        silence_start = None
        t_start       = time.time()

        # VAD wants 512-sample chunks at 16kHz
        vad_chunk = 512

        while True:
            raw = stream.read(vad_chunk, exception_on_overflow=False)
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

    # ── process loop (transcription, off main thread) ─────────────────────────

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

            result.update({
                "mode":       mode,
                "wake_word":  ww_name,
                "confidence": round(score, 4),
            })

            try:
                self._callback(result)
            except Exception as e:
                print(f"[voice] callback error: {e}")


# ── standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hey-zane",    default="wakeword_models/hey_zane.onnx")
    parser.add_argument("--zane-write",  default="wakeword_models/zane_write.onnx")
    parser.add_argument("--threshold",   type=float, default=OWW_THRESHOLD)
    parser.add_argument("--whisper",     default=WHISPER_MODEL)
    args = parser.parse_args()

    def on_result(r):
        print(f"\n{'─'*50}")
        print(f"mode:       {r['mode']}")
        print(f"wake word:  {r['wake_word']}  (conf: {r['confidence']:.3f})")
        print(f"language:   {r['language']}")
        print(f"text:       {r['text']}")
        if r["words"]:
            print("words:")
            for w in r["words"]:
                print(f"  {w['start']:.2f}s–{w['end']:.2f}s  {w['word']!r}  p={w['probability']:.3f}")
        print(f"{'─'*50}\n")

    listener = VoiceListener(
        hey_zane_model=args.hey_zane,
        zane_write_model=args.zane_write,
        callback=on_result,
        oww_threshold=args.threshold,
        whisper_model=args.whisper,
    )

    with listener:
        listener.start()
        print("say 'hey zane' or 'zane write' — ctrl+c to quit\n")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nquitting...")
