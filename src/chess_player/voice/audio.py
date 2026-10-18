"""Microphone listener (arecord + adaptive voice detection + Whisper) and speaker (Piper TTS + aplay).

The laptop's digital mic delivers a large DC offset (about -8000), so frames are high-passed before measuring
loudness. Speech = loudness well above the running noise floor; an utterance ends after VOICE_END_SILENCE_S of
quiet. The mic is ignored while the speaker talks (and a moment after), so the robot never hears itself.
"""
import collections
import io
import queue
import subprocess
import threading
import time
import wave

import numpy as np

from chess_player import config as C

RATE = 16000
FRAME = 480  # 30 ms


class Speaker:
    """Text -> speech in a background thread. say() returns at once; lines play in order."""

    def __init__(self):
        from piper import PiperVoice

        self.voice = PiperVoice.load(str(C.PIPER_VOICE))
        self.q = queue.Queue()
        self.speaking = False
        self.current = ""  # the line being spoken right now
        self.quiet_until = 0.0
        self.muted = False
        self.volume = C.VOICE_VOLUME  # 0 .. 1.5
        self.speed = C.VOICE_SPEED  # 0.6 .. 1.6 (1 = normal)
        self._proc = None
        threading.Thread(target=self._loop, daemon=True).start()

    def say(self, text):
        if text and not self.muted:
            self.q.put(text)

    def stop(self):
        """Stop the current line and drop anything queued."""
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        if self._proc and self._proc.poll() is None:
            self._proc.kill()

    def _wav(self, text):
        from piper import SynthesisConfig

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            cfg = SynthesisConfig(length_scale=1.0 / max(0.3, self.speed), volume=max(0.0, min(1.5, self.volume)))
            self.voice.synthesize_wav(text, w, syn_config=cfg)
        return buf.getvalue()

    def _loop(self):
        while True:
            text = self.q.get()
            try:
                data = self._wav(text)
                self.speaking, self.current = True, text
                self._proc = subprocess.Popen(["aplay", "-q", "-"], stdin=subprocess.PIPE)
                self._proc.communicate(data)
            except Exception as e:
                print(f"[voice] speak failed: {e}", flush=True)
            finally:
                self.speaking, self.current = False, ""
                self.quiet_until = time.time() + 0.5


class Transcriber:
    def __init__(self):
        import torch
        from transformers import pipeline

        self.pipe = pipeline("automatic-speech-recognition", model=C.WHISPER_MODEL,
                             device=0 if torch.cuda.is_available() else -1,
                             dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

    def __call__(self, audio_f32):
        out = self.pipe({"raw": audio_f32, "sampling_rate": RATE},
                        generate_kwargs={"language": C.VOICE_LANGUAGE, "task": "transcribe"})
        return out["text"].strip()


class Listener:
    """Calls on_text(text) for every utterance heard (while not muted / not speaking)."""

    def __init__(self, on_text, speaker=None, transcriber=None):
        self.on_text, self.speaker = on_text, speaker
        self.stt = transcriber or Transcriber()
        self.state = "listening"  # listening | hearing | transcribing | off
        self.enabled = C.VOICE_MIC_ON_AT_START
        self.level = 0.0
        self.threshold = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                self._run()
            except Exception as e:
                print(f"[voice] mic error: {e}; retrying", flush=True)
            self.state = "off"
            time.sleep(2.0)

    def _run(self):
        cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw"]
        if C.VOICE_MIC_DEVICE:
            cmd += ["-D", C.VOICE_MIC_DEVICE]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        floor = collections.deque(maxlen=200)  # ~6 s of frame loudness for the noise floor
        pre = collections.deque(maxlen=10)  # 300 ms before speech starts
        dc = 0.0
        voiced, silent, utter = 0, 0, []
        self.state = "listening"
        try:
            while True:
                raw = p.stdout.read(FRAME * 2)
                if len(raw) < FRAME * 2:
                    raise RuntimeError("arecord stopped")
                x = np.frombuffer(raw, np.int16).astype(np.float32)
                dc = 0.995 * dc + 0.005 * float(x.mean()) if dc else float(x.mean())
                x -= dc
                rms = float(np.sqrt(np.mean(x * x)))
                self.level = rms
                busy = (self.speaker is not None and (self.speaker.speaking or time.time() < self.speaker.quiet_until))
                if not self.enabled or busy:
                    voiced, silent, utter = 0, 0, []
                    pre.clear()
                    continue
                if not utter:
                    floor.append(rms)
                thr = max(C.VOICE_MIN_RMS, C.VOICE_SNR * (np.percentile(floor, 20) if len(floor) > 20 else 300.0))
                self.threshold = thr
                loud = rms > thr
                if not utter:
                    pre.append(x)
                    voiced = voiced + 1 if loud else 0
                    if voiced >= 3:  # 90 ms of speech: start an utterance
                        utter = list(pre)
                        self.state = "hearing"
                        silent = 0
                    continue
                utter.append(x)
                silent = 0 if loud else silent + 1
                dur = len(utter) * FRAME / RATE
                if silent * FRAME / RATE >= C.VOICE_END_SILENCE_S or dur >= C.VOICE_MAX_UTTERANCE_S:
                    audio = np.concatenate(utter) / 32768.0
                    utter, voiced = [], 0
                    pre.clear()
                    if dur >= C.VOICE_MIN_UTTERANCE_S:
                        self.state = "transcribing"
                        text = self.stt(np.clip(audio, -1, 1).astype(np.float32))
                        if text and len(text) > 1:
                            self.on_text(text)
                    self.state = "listening"
        finally:
            p.kill()
