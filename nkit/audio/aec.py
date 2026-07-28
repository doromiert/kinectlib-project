"""
nkit/audio/aec.py — calibratable self-sound cancellation (acoustic echo cancellation)

ctypes wrapper around libspeexdsp's echo canceller + preprocessor (residual
echo suppression), same "wrap the C lib with ctypes" pattern this repo
already uses for libfreenect2 (see kinect.py / kinect_shim.cpp). Needs
`pkgs.speexdsp` on LD_LIBRARY_PATH — see flake.nix.

Two pieces:

  calibrate() — one-time-per-setup routine. Plays a short tone through the
  OS's default output, records it back via BOTH the mic and the system
  loopback/monitor source (PipeWire/Pulse `*.monitor` — this is what
  becomes the AEC reference signal at runtime, since it captures whatever
  the OS actually sent to the speaker, post-volume/post-mixing), and
  cross-correlates the two to find the delay + gain between them. Re-run
  if audio hardware changes.

  EchoCanceller — the runtime piece. Continuously fed reference audio from
  the monitor source (push_reference()) and mic audio to clean (process()).
  Internally delays the reference stream by the calibrated delay before
  handing (mic, reference) pairs to speexdsp, since the monitor tap and the
  mic don't hear the same moment in time.

Caveat: the calibration routine and the runtime delay-alignment logic
haven't been run against real hardware yet (no Kinect/audio devices in the
dev environment this was written in) — the frame sizes and thresholds are
principled defaults, not hardware-verified. Expect to tune
CALIBRATION_TONE_* / AEC_FRAME_SIZE once you can test on the real box.
"""

from __future__ import annotations
import ctypes
import ctypes.util
import numpy as np

from ..types import AecCalibration

SAMPLE_RATE = 16000
AEC_FRAME_SIZE = 512          # 32ms @ 16kHz — matches silero VAD's hard 512-sample requirement
FILTER_LENGTH_MS = 200        # how much echo tail (ms) the canceller can cancel

CALIBRATION_TONE_FREQ_HZ = 1000.0
CALIBRATION_TONE_DURATION_S = 1.5
CALIBRATION_MAX_DELAY_MS = 500.0

# speex_echo.h
SPEEX_ECHO_SET_SAMPLING_RATE = 24

# speex_preprocess.h
SPEEX_PREPROCESS_SET_DENOISE        = 0
SPEEX_PREPROCESS_SET_AGC            = 2
SPEEX_PREPROCESS_SET_NOISE_SUPPRESS = 18
SPEEX_PREPROCESS_SET_ECHO_STATE     = 24


def _load_lib():
    name = ctypes.util.find_library("speexdsp")
    candidates = [name] if name else []
    candidates += ["libspeexdsp.so.1", "libspeexdsp.so"]
    last_err = None
    for c in candidates:
        if not c:
            continue
        try:
            return ctypes.CDLL(c)
        except OSError as e:
            last_err = e
    raise RuntimeError(
        "libspeexdsp not found — add pkgs.speexdsp to the nix devShell "
        "(flake.nix) so it's on LD_LIBRARY_PATH"
    ) from last_err


lib = _load_lib()

lib.speex_echo_state_init.restype  = ctypes.c_void_p
lib.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]

lib.speex_echo_state_destroy.restype  = None
lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]

lib.speex_echo_cancellation.restype  = None
lib.speex_echo_cancellation.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int16),
    ctypes.POINTER(ctypes.c_int16),
    ctypes.POINTER(ctypes.c_int16),
]

lib.speex_echo_ctl.restype  = ctypes.c_int
lib.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

lib.speex_preprocess_state_init.restype  = ctypes.c_void_p
lib.speex_preprocess_state_init.argtypes = [ctypes.c_int, ctypes.c_int]

lib.speex_preprocess_state_destroy.restype  = None
lib.speex_preprocess_state_destroy.argtypes = [ctypes.c_void_p]

lib.speex_preprocess_run.restype  = ctypes.c_int
lib.speex_preprocess_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16)]

lib.speex_preprocess_ctl.restype  = ctypes.c_int
lib.speex_preprocess_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


# ── device discovery (shared by calibrate() and voice.py) ──────────────────────

def _find_device(pa, name_substrings: list[str], is_input: bool) -> int | None:
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        channels = info["maxInputChannels"] if is_input else info["maxOutputChannels"]
        if channels < 1:
            continue
        name = info["name"].lower()
        if any(s in name for s in name_substrings):
            return i
    return None


def find_kinect_mic(pa) -> int:
    idx = _find_device(pa, ["kinect", "xbox", "nui"], is_input=True)
    return idx if idx is not None else pa.get_default_input_device_info()["index"]


def find_monitor_source(pa) -> int | None:
    """PipeWire/PulseAudio loopback capture device — 'Monitor of ...' / '*.monitor'."""
    return _find_device(pa, ["monitor"], is_input=True)


# ── delayed reference ring buffer ───────────────────────────────────────────────

class _DelayedReferenceBuffer:
    """
    Ring buffer of reference (loopback/monitor) samples, read back delayed
    by the calibrated delay so it lines up with what the mic captures for
    the same moment in time. The monitor tap hears audio before it
    physically/acoustically reaches the mic, so the reference has to be
    held back by that gap before pairing it with a mic frame.
    """

    def __init__(self, delay_samples: int, frame_size: int):
        self._delay = max(0, delay_samples)
        self._frame_size = frame_size
        self._buf = np.zeros(self._delay + frame_size * 8, dtype=np.int16)
        self._write_pos = 0
        self._filled = 0

    def push(self, chunk: np.ndarray) -> None:
        n = len(chunk)
        cap = len(self._buf)
        end = self._write_pos + n
        if end <= cap:
            self._buf[self._write_pos:end] = chunk
        else:
            first = cap - self._write_pos
            self._buf[self._write_pos:] = chunk[:first]
            self._buf[:end - cap] = chunk[first:]
        self._write_pos = end % cap
        self._filled = min(cap, self._filled + n)

    def pop_delayed(self) -> np.ndarray | None:
        """Next frame_size samples from `delay` behind the write head, or None if not enough history yet."""
        needed = self._delay + self._frame_size
        if self._filled < needed:
            return None
        cap = len(self._buf)
        start = (self._write_pos - needed) % cap
        idx = (start + np.arange(self._frame_size)) % cap
        return self._buf[idx]


# ── runtime echo canceller ──────────────────────────────────────────────────────

class EchoCanceller:
    def __init__(
        self,
        calibration: AecCalibration,
        frame_size: int = AEC_FRAME_SIZE,
        sample_rate: int = SAMPLE_RATE,
        filter_length_ms: int = FILTER_LENGTH_MS,
        denoise: bool = True,
        noise_suppress_db: int = -15,
    ):
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self._gain = calibration.gain

        filter_length = int(sample_rate * filter_length_ms / 1000)
        self._echo_state = lib.speex_echo_state_init(frame_size, filter_length)
        if not self._echo_state:
            raise RuntimeError("speex_echo_state_init failed")

        rate = ctypes.c_int(sample_rate)
        lib.speex_echo_ctl(self._echo_state, SPEEX_ECHO_SET_SAMPLING_RATE, ctypes.byref(rate))

        self._preprocess_state = lib.speex_preprocess_state_init(frame_size, sample_rate)
        if not self._preprocess_state:
            lib.speex_echo_state_destroy(self._echo_state)
            raise RuntimeError("speex_preprocess_state_init failed")

        denoise_flag = ctypes.c_int(1 if denoise else 0)
        lib.speex_preprocess_ctl(self._preprocess_state, SPEEX_PREPROCESS_SET_DENOISE, ctypes.byref(denoise_flag))
        suppress = ctypes.c_int(noise_suppress_db)
        lib.speex_preprocess_ctl(self._preprocess_state, SPEEX_PREPROCESS_SET_NOISE_SUPPRESS, ctypes.byref(suppress))
        # SET_ECHO_STATE takes the echo state pointer itself as the "ptr" arg, not a pointer to it
        lib.speex_preprocess_ctl(self._preprocess_state, SPEEX_PREPROCESS_SET_ECHO_STATE, self._echo_state)

        delay_samples = int(calibration.delay_ms / 1000.0 * sample_rate)
        self._ref_buf = _DelayedReferenceBuffer(delay_samples, frame_size)

    def push_reference(self, chunk: bytes) -> None:
        """Feed a chunk (raw int16 bytes, any length) captured from the loopback/monitor source."""
        arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        arr = np.clip(arr * self._gain, -32768, 32767).astype(np.int16)
        self._ref_buf.push(arr)

    def process(self, mic_chunk: bytes) -> bytes:
        """
        mic_chunk: exactly frame_size int16 samples (raw bytes).
        Returns cleaned int16 PCM bytes, same length — passes mic audio
        through unmodified if there isn't enough reference history yet
        (first ~calibration.delay_ms of runtime).
        """
        ref = self._ref_buf.pop_delayed()
        if ref is None:
            return mic_chunk

        rec  = (ctypes.c_int16 * self.frame_size).from_buffer_copy(mic_chunk)
        play = (ctypes.c_int16 * self.frame_size)(*ref.tolist())
        out  = (ctypes.c_int16 * self.frame_size)()
        lib.speex_echo_cancellation(self._echo_state, rec, play, out)
        lib.speex_preprocess_run(self._preprocess_state, out)
        return bytes(out)

    def close(self):
        if self._echo_state:
            lib.speex_echo_state_destroy(self._echo_state)
            self._echo_state = None
        if self._preprocess_state:
            lib.speex_preprocess_state_destroy(self._preprocess_state)
            self._preprocess_state = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── calibration ──────────────────────────────────────────────────────────────

def calibrate(
    mic_index: int | None = None,
    monitor_index: int | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> AecCalibration:
    """
    Plays a short tone through the OS default output, records it back via
    both the mic and the loopback/monitor source, cross-correlates to find
    the delay + gain between them. Run once per audio setup.
    """
    import pyaudio

    pa = pyaudio.PyAudio()
    out_stream = mic_stream = mon_stream = None
    try:
        if mic_index is None:
            mic_index = find_kinect_mic(pa)
        if monitor_index is None:
            monitor_index = find_monitor_source(pa)
            if monitor_index is None:
                raise RuntimeError(
                    "no PipeWire/Pulse monitor source found — is a '*.monitor' "
                    "device exposed? it's needed as the AEC reference signal"
                )

        n_tone = int(sample_rate * CALIBRATION_TONE_DURATION_S)
        t = np.arange(n_tone) / sample_rate
        tone = (np.sin(2 * np.pi * CALIBRATION_TONE_FREQ_HZ * t) * 0.5 * 32767).astype(np.int16)

        capture_frames = 1024
        n_capture = n_tone + int(sample_rate * CALIBRATION_MAX_DELAY_MS / 1000)

        out_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True)
        mic_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True,
                              input_device_index=mic_index, frames_per_buffer=capture_frames)
        mon_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True,
                              input_device_index=monitor_index, frames_per_buffer=capture_frames)

        import threading

        mic_chunks: list[bytes] = []
        mon_chunks: list[bytes] = []

        def _record(stream, chunks, n_samples):
            got = 0
            while got < n_samples:
                chunks.append(stream.read(capture_frames, exception_on_overflow=False))
                got += capture_frames

        t_mic = threading.Thread(target=_record, args=(mic_stream, mic_chunks, n_capture))
        t_mon = threading.Thread(target=_record, args=(mon_stream, mon_chunks, n_capture))
        t_mic.start()
        t_mon.start()
        out_stream.write(tone.tobytes())
        t_mic.join()
        t_mon.join()

        mic_audio = np.frombuffer(b"".join(mic_chunks), dtype=np.int16).astype(np.float32)
        mon_audio = np.frombuffer(b"".join(mon_chunks), dtype=np.int16).astype(np.float32)
        n = min(len(mic_audio), len(mon_audio))
        mic_audio, mon_audio = mic_audio[:n], mon_audio[:n]

        corr = np.correlate(mic_audio, mon_audio, mode="full")
        lag_samples = int(np.argmax(corr)) - (n - 1)
        delay_ms = max(0.0, lag_samples / sample_rate * 1000.0)

        mic_rms = float(np.sqrt(np.mean(mic_audio ** 2)) + 1e-9)
        mon_rms = float(np.sqrt(np.mean(mon_audio ** 2)) + 1e-9)
        gain = mic_rms / mon_rms

        norm = float(np.linalg.norm(mic_audio) * np.linalg.norm(mon_audio) + 1e-9)
        quality = float(np.clip(np.max(corr) / norm, 0.0, 1.0))

        return AecCalibration(delay_ms=round(delay_ms, 2), gain=round(gain, 4), quality_score=round(quality, 4))
    finally:
        for s in (out_stream, mic_stream, mon_stream):
            if s is not None:
                try:
                    s.stop_stream()
                    s.close()
                except Exception:
                    pass
        pa.terminate()
