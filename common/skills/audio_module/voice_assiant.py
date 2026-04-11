#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
English Voice Assistant with Doorbell Detection (Local YAMNet)
Supports manual Enter key fallback when no doorbell detected.
"""

import time
import threading
import collections
import requests
import io
import pygame
import numpy as np
import noisereduce as nr
import os
import tempfile
import re
import hashlib
import subprocess
import pyaudio
import wave
import webrtcvad
import spacy
import sys
import select

# ------------------ Doorbell detection (Local YAMNet) ------------------
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
YAMNET_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yamnet")

# ------------------ Configuration ------------------
ASR_URL = "http://127.0.0.1:8001/api/speech_recognition"
TTS_URL = "http://127.0.0.1:8002/v1/audio/speech"

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION_MS = 30
CHUNK = int(RATE * FRAME_DURATION_MS / 1000)
SPEECH_START_FRAMES = 12
SILENCE_TIMEOUT_MS = 1500
MAX_RECORD_DURATION_MS = 10000
RING_BUFFER_MAXLEN = int(1000 / FRAME_DURATION_MS)
MAX_LOW_ENERGY_FRAMES = 30

TTS_CACHE_DIR = "tts_cache"
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

OFFLINE_TTS_CMD = "pico2wave"
OFFLINE_TTS_BACKUP = "espeak"

COMMON_DRINKS_EN = [
    "coke", "coca cola", "pepsi", "sprite", "fanta", "7up", "juice", "orange juice",
    "apple juice", "milk", "yogurt", "water", "mineral water", "tea", "black tea",
    "green tea", "oolong tea", "coffee", "latte", "cappuccino", "beer", "wine",
    "red wine", "white wine", "cocktail", "lemonade", "soda"
]

# Load English NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("English model not found. Installing...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# ------------------ Helper Functions ------------------
def extract_name(text):
    if not text:
        return None
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    patterns = [
        r"(?:my name is|i am|i'm|called?|name's?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"^(?:i'?m?|this is)\s+([A-Z][a-z]+)",
        r"call me\s+([A-Z][a-z]+)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1)
    words = re.findall(r'[A-Z][a-z]+', text)
    return words[0] if words else None

def extract_drink(text):
    if not text:
        return None
    text_lower = text.lower()
    for drink in COMMON_DRINKS_EN:
        if drink in text_lower:
            return drink.capitalize()
    return None
def wait_for_doorbell_with_fallback(detector):
    print("Waiting for doorbell... (Press Enter to manually trigger)")
    while True:
        if detector and detector.wait_for_doorbell(timeout=0.5):
            print("Doorbell detected!")
            return True
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if line.strip() == "":
                print("Manual trigger by Enter key.")
                return True
def ask_info(assistant, prompt, extract_func, retry_prompt, default=None, max_attempts=2):
    for attempt in range(max_attempts):
        assistant.speak(prompt if attempt == 0 else retry_prompt)
        audio = assistant.record()
        if audio:
            text = assistant.recognize(audio)
            if text:
                info = extract_func(text)
                if info:
                    return info
    return default
# ------------------ Doorbell Detector Class (Local Model) ------------------
class DoorbellDetector:
    def __init__(self, threshold=0.5, chunk_seconds=1.0):
        self.threshold = threshold
        self.chunk_seconds = chunk_seconds
        self.running = False
        self.doorbell_detected = threading.Event()
        self.stream = None
        self.p = None
        self.model = None
        self.doorbell_index = None
        self._init_model()

    def _init_model(self):
        print("[Doorbell] Loading YAMNet model from:", YAMNET_MODEL_PATH)
        self.model = tf.saved_model.load(YAMNET_MODEL_PATH)

        csv_path = os.path.join(YAMNET_MODEL_PATH, 'assets', 'yamnet_class_map.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Class map file not found: {csv_path}")

        class_names = []
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    class_names.append(parts[2].strip('"'))

        for i, name in enumerate(class_names):
            if 'doorbell' in name.lower():
                self.doorbell_index = i
                break

        if self.doorbell_index is None:
            raise ValueError("Doorbell class not found in class map")
        print(f"[Doorbell] Doorbell class index: {self.doorbell_index} ({class_names[self.doorbell_index]})")

    def _detect(self, audio_bytes):
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        scores, embeddings, spectrogram = self.model(audio)
        doorbell_probs = scores[:, self.doorbell_index].numpy()
        avg_prob = np.mean(doorbell_probs)
        return avg_prob > self.threshold

    def start(self):
        if self.running:
            return
        self.running = True
        self.doorbell_detected.clear()
        self.p = pyaudio.PyAudio()
        chunk_size = int(RATE * self.chunk_seconds)
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                  input=True, frames_per_buffer=chunk_size)
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()
        print("[Doorbell] Listening started...")

    def _listen(self):
        while self.running:
            try:
                data = self.stream.read(int(RATE * self.chunk_seconds), exception_on_overflow=False)
                if self._detect(data):
                    print("[Doorbell] Doorbell detected!")
                    self.doorbell_detected.set()
                    time.sleep(2)
            except Exception as e:
                print(f"[Doorbell] Error: {e}")
                time.sleep(0.5)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        print("[Doorbell] Listening stopped.")

    def wait_for_doorbell(self, timeout=None):
        return self.doorbell_detected.wait(timeout)

# ------------------ VoiceAssistant Class ------------------
class VoiceAssistant:
    def __init__(self):
        self.vad = webrtcvad.Vad(3)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.noise_floor = 500.0
        self.energy_history = collections.deque(maxlen=50)
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        self.offline_tts_available = self._check_offline_tts()

    def _check_offline_tts(self):
        try:
            subprocess.run([OFFLINE_TTS_CMD, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1)
            print("[Info] Offline TTS (pico2wave) available")
            return True
        except:
            try:
                subprocess.run([OFFLINE_TTS_BACKUP, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1)
                print("[Info] Offline TTS (espeak) available")
                return True
            except:
                print("[Warning] No offline TTS available")
                return False

    def _offline_tts(self, text):
        if not self.offline_tts_available:
            return False
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            if subprocess.run([OFFLINE_TTS_CMD, "-l", "en-US", "-w", temp_path, text], check=False, timeout=5).returncode == 0:
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                time.sleep(0.3)
                os.unlink(temp_path)
                return True
            if subprocess.run([OFFLINE_TTS_BACKUP, "-v", "en-us", "-w", temp_path, text], check=False, timeout=5).returncode == 0:
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                time.sleep(0.3)
                os.unlink(temp_path)
                return True
        except Exception as e:
            print(f"[Offline TTS error] {e}")
        return False

    def _get_cache_path(self, text):
        return os.path.join(TTS_CACHE_DIR, hashlib.md5(text.encode()).hexdigest() + ".mp3")

    def preload_tts(self, text):
        cache_path = self._get_cache_path(text)
        if os.path.exists(cache_path):
            return
        try:
            payload = {"model": "tts-1", "input": text, "voice": "alloy", "speed": 1.0}
            resp = requests.post(TTS_URL, json=payload, timeout=10)
            if resp.status_code == 200:
                with open(cache_path, 'wb') as f:
                    f.write(resp.content)
                print(f"[Preload] Cached: {text[:50]}...")
        except Exception as e:
            print(f"[Preload] Error: {e}")

    def preload_common_phrases(self):
        common_texts = [
            "Hello, welcome to the party! What's your name?",
            "Sorry, I didn't catch your name. Could you say it again?",
            "What's your favorite drink?",
            "Sorry, I didn't catch that. What's your favorite drink?",
            "Great. Please have a seat.",
            "Thank you both for coming! Enjoy the party."
        ]
        for text in common_texts:
            threading.Thread(target=self.preload_tts, args=(text,), daemon=True).start()

    def set_recording(self, state):
        if state == 1:
            if self.stream is None:
                self.stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                              input=True, frames_per_buffer=CHUNK)
                time.sleep(0.3)
                for _ in range(5):
                    try:
                        self.stream.read(CHUNK, exception_on_overflow=False)
                    except:
                        pass
        elif state == 0:
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
        elif state == 2:
            if self.stream is not None and self.stream.is_active():
                self.stream.stop_stream()
        elif state == 3:
            if self.stream is not None and not self.stream.is_active():
                self.stream.start_stream()
                for _ in range(5):
                    try:
                        self.stream.read(CHUNK, exception_on_overflow=False)
                    except:
                        pass

    def speak(self, text):
        print(f"[Robot]: {text}")
        self.set_recording(2)
        cache_path = self._get_cache_path(text)
        if os.path.exists(cache_path):
            try:
                pygame.mixer.music.load(cache_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                time.sleep(0.3)
                self.set_recording(3)
                return
            except:
                pass
        try:
            resp = requests.post(TTS_URL, json={"model": "tts-1", "input": text, "voice": "alloy", "speed": 1.0}, timeout=5)
            if resp.status_code == 200:
                with open(cache_path, 'wb') as f:
                    f.write(resp.content)
                pygame.mixer.music.load(io.BytesIO(resp.content))
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                time.sleep(0.3)
                self.set_recording(3)
                return
        except:
            pass
        if self._offline_tts(text):
            self.set_recording(3)
            return
        self.set_recording(3)
        print(f"[TTS Failed] Could not play: {text}")

    def update_noise_floor(self, frame_energy, is_speech_vad):
        if not is_speech_vad:
            self.energy_history.append(frame_energy)
            if len(self.energy_history) >= 10:
                sorted_energies = sorted(self.energy_history)
                p20_index = max(1, int(len(sorted_energies) * 0.2)) - 1
                self.noise_floor = min(max(sorted_energies[p20_index] * 1.2, 20.0), 8000.0)

    def denoise(self, audio_frames):
        if not audio_frames:
            return audio_frames
        try:
            audio_bytes = b''.join(audio_frames)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            noise_sample_len = int(0.5 * RATE)
            noise_sample = audio_float32[:min(noise_sample_len, len(audio_float32))]
            reduced = nr.reduce_noise(y=audio_float32, sr=RATE, y_noise=noise_sample if len(noise_sample) > 0 else None,
                                      prop_decrease=1.0, time_constant_s=2.0, freq_mask_smooth_hz=500,
                                      time_mask_smooth_ms=50, thresh_n_mult_nonstationary=2,
                                      sigmoid_slope_nonstationary=10, n_std_thresh_stationary=1.5,
                                      stationary=False, n_fft=1024, win_length=1024, hop_length=256)
            reduced_int16 = (reduced * 32768).astype(np.int16)
            frame_bytes = reduced_int16.tobytes()
            frame_size = CHUNK * 2
            return [frame_bytes[i:i+frame_size] for i in range(0, len(frame_bytes), frame_size) if len(frame_bytes[i:i+frame_size]) == frame_size]
        except:
            return audio_frames

    def _cleanup_temp_files(self, max_files=50):
        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        try:
            files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.wav')]
            if len(files) > max_files:
                files.sort(key=os.path.getmtime)
                for f in files[:-max_files]:
                    os.remove(f)
        except:
            pass

    def recognize(self, audio_frames):
        denoised = self.denoise(audio_frames)
        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        temp_path = os.path.join(temp_dir, f"temp_{timestamp}.wav")
        wf = wave.open(temp_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(denoised))
        wf.close()
        try:
            with open(temp_path, 'rb') as af:
                resp = requests.post(ASR_URL, files={"audio": ("audio.wav", af, "audio/wav")}, timeout=5)
                if resp.status_code == 200:
                    text = resp.json().get('text', '').strip()
                    text = re.sub(r'<\|[^>]+\|>', '', text)
                    text = re.sub(r'\b(uh|um|ah|eh|er|mm|hmm)\b', '', text, flags=re.IGNORECASE)
                    text = re.sub(r'[^\w\s\']', '', text)
                    return text.strip()
        except Exception as e:
            print(f"[ASR error] {e}")
        finally:
            self._cleanup_temp_files(50)
            try:
                os.unlink(temp_path)
            except:
                pass
        return ""

    def record(self):
        if self.stream is None:
            return []
        ring_buffer = collections.deque(maxlen=RING_BUFFER_MAXLEN)
        frames = []
        recording = False
        silence_frames = 0
        speech_frames = 0
        low_energy_frames = 0
        total_frames = 0
        max_silence_frames = int(SILENCE_TIMEOUT_MS / FRAME_DURATION_MS)
        max_total_frames = int(MAX_RECORD_DURATION_MS / FRAME_DURATION_MS)
        while True:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            audio_array = np.frombuffer(data, dtype=np.int16)
            energy = np.sqrt(np.mean(audio_array.astype(float)**2))
            is_vad = self.vad.is_speech(data, RATE)
            self.update_noise_floor(energy, is_vad)
            thresh = self.noise_floor * 1.2
            is_speech = is_vad and (energy > thresh)
            total_frames += 1
            if not recording:
                ring_buffer.append(data)
                if is_speech:
                    speech_frames += 1
                    if speech_frames >= SPEECH_START_FRAMES:
                        frames.extend(ring_buffer)
                        recording = True
                else:
                    speech_frames = 0
            else:
                frames.append(data)
                if is_speech:
                    silence_frames = 0
                    low_energy_frames = 0
                else:
                    silence_frames += 1
                    low_energy_frames += 1
                    if silence_frames >= max_silence_frames or low_energy_frames >= MAX_LOW_ENERGY_FRAMES:
                        break
            if total_frames >= max_total_frames:
                break
        return frames

    def close(self):
        self.set_recording(0)
        self.audio.terminate()
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()

# ------------------ Compatibility aliases for state machine ------------------
VoiceAssistant.record_utterance = VoiceAssistant.record

# 创建全局实例（单例）
voice_assistant = VoiceAssistant()
doorbell = DoorbellDetector()

# 为兼容状态机框架的导入接口
def get_voice_assistant():
    return voice_assistant

def create_doorbell_detector(threshold=0.5, chunk_seconds=1.0):
    return DoorbellDetector(threshold, chunk_seconds)

def main():
    # 初始化门铃检测器（可选，若不需要可设为 None）
    try:
        doorbell = DoorbellDetector(threshold=0.5)
        doorbell.start()
    except Exception as e:
        print(f"门铃检测器启动失败: {e}，将使用手动触发")
        doorbell = None

    assistant = VoiceAssistant()
    assistant.set_recording(0)  # 初始时关闭录音流

    # 等待第一位客人
    print("等待门铃... (按 Enter 手动触发)")
    if doorbell:
        wait_for_doorbell_with_fallback(doorbell)
    else:
        input("按 Enter 模拟门铃...")

    assistant.set_recording(1)

    name1 = ask_info(assistant,
                     "Hello, welcome to the party! What's your name?",
                     extract_name,
                     "Sorry, I didn't catch your name. Could you say it again?",
                     "Friend")
    drink1 = ask_info(assistant,
                      f"{name1}, what's your favorite drink?",
                      extract_drink,
                      "Sorry, I didn't catch that. What's your favorite drink?",
                      "Water")
    assistant.speak(f"Great, {name1}. Please have a seat.")

    assistant.set_recording(0)

    # 等待第二位客人
    print("等待第二位客人门铃... (按 Enter 手动触发)")
    if doorbell:
        wait_for_doorbell_with_fallback(doorbell)
    else:
        input("按 Enter 模拟门铃...")

    assistant.set_recording(1)

    name2 = ask_info(assistant,
                     "Hello, welcome to the party! What's your name?",
                     extract_name,
                     "Sorry, I didn't catch your name. Could you say it again?",
                     "Friend")
    drink2 = ask_info(assistant,
                      f"{name2}, what's your favorite drink?",
                      extract_drink,
                      "Sorry, I didn't catch that. What's your favorite drink?",
                      "Water")
    assistant.speak(f"Great, {name2}. Please have a seat.")

    assistant.speak(f"{name2}, let me introduce {name1}, who likes to drink {drink1}.")
    assistant.speak(f"{name1}, this is {name2}, who likes to drink {drink2}.")
    assistant.speak("Thank you both for coming! Enjoy the party.")

    assistant.close()
    if doorbell:
        doorbell.stop()
    print("程序结束。")

if __name__ == "__main__":
    main()