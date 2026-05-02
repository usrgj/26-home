#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整语音助手模块 - 支持全流程测试（两位客人姓名&饮品）
优化名字提取，增强 Richard 和 Jennier 等名字的识别纠正。
直接运行本脚本即可开始测试，无需机器人硬件。
"""

import time
import threading
import collections
import requests
import numpy as np
import os
import tempfile
import re
import hashlib
import subprocess
import wave
import webrtcvad
import spacy
import sys
import select
from difflib import get_close_matches
from pathlib import Path
import pyaudio

# ---------- 配置 ----------
DOORBELL_ENABLED = True          # 测试时可保留，但流程中会跳过
INPUT_DEVICE_INDEX = None
RECORD_DURATION = 1.5
SIMILARITY_THRESHOLD = 0.75
SMOOTHING_WINDOW = 5
COOLDOWN_SECONDS = 0.5

# TTS 语速（1.0 正常，1.2 略快）
TTS_SPEED = 1.2

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from common.config import LANGUAGE, COMMON_DRINKS

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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TTS_CACHE_DIR = os.path.join(CURRENT_DIR, "audio_cache")
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

# 饮料列表
if LANGUAGE == "en":
    COMMON_DRINKS_LIST = COMMON_DRINKS
else:
    COMMON_DRINKS_LIST = [
        "可乐", "雪碧", "芬达", "美年达", "七喜", "果汁", "橙汁", "苹果汁",
        "牛奶", "酸奶", "水", "矿泉水", "茶", "红茶", "绿茶", "乌龙茶",
        "咖啡", "拿铁", "卡布奇诺", "啤酒", "红酒"
    ]

# 加载英文 NLP
if LANGUAGE == "en":
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
else:
    nlp = None

# ---------- 辅助函数 ----------
def safe_filename(text: str, max_len=100) -> str:
    """将文本转换为安全、可读的文件名"""
    illegal_chars = r'[\\/*?:"<>|]'
    safe = re.sub(illegal_chars, '_', text)
    safe = safe.strip('. ')
    if len(safe) > max_len:
        safe = safe[:max_len]
    return safe + ".mp3"

# ---------- 名字提取优化 ----------
def _normalize_name(name: str) -> str:
    """将识别出的名字纠正为标准形式（针对英文）"""
    if not name:
        return name
    # 标准名字列表（首字母大写）
    known_names = ["Jack", "John", "Richard", "Allen", "Mike", "Grace", "Linda", "Lily", "Lucy", "Jennier"]
    # 常见错误映射（不区分大小写）
    correction_map = {
        "rechard": "Richard",
        "richord": "Richard",
        "recard": "Richard",
        "jenny": "Jennier",
        "jennie": "Jennier",
        "jennifer": "Jennier",
        "gennifer": "Jennier",
        "jenn": "Jennier",
        "jhon": "John",
        "jon": "John",
        "mikey": "Mike",
        "lucyy": "Lucy",
        "lindaa": "Linda",
        "gracy": "Grace",
    }
    lower_name = name.lower()
    # 精确匹配或前缀匹配
    for wrong, correct in correction_map.items():
        if lower_name == wrong or lower_name.startswith(wrong):
            return correct
    # 模糊匹配最接近的标准名字
    matches = get_close_matches(lower_name, [n.lower() for n in known_names], n=1, cutoff=0.6)
    if matches:
        for kn in known_names:
            if kn.lower() == matches[0]:
                return kn
    # 如果识别结果本身就在标准列表中（或首字母大写形式），直接返回
    if name in known_names or name.capitalize() in known_names:
        return name.capitalize()
    return name

def extract_name_en(text):
    if not text:
        return None

    # 1. 转为小写用于前缀匹配（但保留原大小写用于最后返回）
    lower_text = text.lower()
    cleaned = text  # 默认原始文本

    # 2. 常见前缀列表（按长度从长到短排序，避免误匹配）
    prefixes = [
        "my name is ", "i am ", "i'm ", "name is ", "this is ", "call me ",
        "i m ", "im ", "un ", "um ", "uh ", "er ", "and ", "so "
    ]
    for prefix in prefixes:
        if lower_text.startswith(prefix):
            cleaned = text[len(prefix):]
            break

    # 3. 如果清理后的文本以 "un " 开头（针对 "Un Jack" 情况），再次去除
    if cleaned.lower().startswith("un "):
        cleaned = cleaned[3:]

    # 4. 使用 spaCy 实体识别
    doc = nlp(cleaned)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            if len(name) >= 2 and name[0].isupper():
                return _normalize_name(name)

    # 5. 正则匹配明确的人名模式
    patterns = [
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",  # 匹配一个或两个大写开头的单词
    ]
    for p in patterns:
        m = re.search(p, cleaned)
        if m:
            name = m.group(1).strip()
            if len(name) >= 2 and name[0].isupper():
                return _normalize_name(name)

    # 6. 后备：提取所有大写开头的单词
    words = re.findall(r'[A-Z][a-z]+', cleaned)
    if words:
        # 如果有多个人名部分，将前两个组合（如 "Jack Smith"）
        if len(words) >= 2 and len(words[0]) > 1 and len(words[1]) > 1:
            name = f"{words[0]} {words[1]}"
        else:
            name = words[0]
        return _normalize_name(name)

    return None

def extract_name_zh(text):
    if not text:
        return None
    patterns = [
        r'(?:我叫|我是|名字是|姓名是|本人叫|本人是)\s*([^\s，。、]+)',
        r'叫\s*([^\s，。、]+)',
        r'^([^\s，。、]{2,4})$',
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)
    words = re.findall(r'[\u4e00-\u9fff]+', text)
    return words[-1] if words else None

def extract_drink_en(text):
    if not text:
        return None
    tl = text.lower()
    for d in COMMON_DRINKS_LIST:
        if d in tl:
            return d.capitalize()
    matches = get_close_matches(tl, COMMON_DRINKS_LIST, n=1, cutoff=0.6)
    return matches[0].capitalize() if matches else None

def extract_drink_zh(text):
    if not text:
        return None
    for d in COMMON_DRINKS_LIST:
        if d in text:
            return d
    matches = get_close_matches(text, COMMON_DRINKS_LIST, n=1, cutoff=0.6)
    return matches[0] if matches else None

if LANGUAGE == "en":
    extract_name = extract_name_en
    extract_drink = extract_drink_en
else:
    extract_name = extract_name_zh
    extract_drink = extract_drink_zh

# ---------- VoiceAssistant ----------
class VoiceAssistant:
    def __init__(self):
        self.vad = webrtcvad.Vad(3)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.noise_floor = 500.0
        self.energy_history = collections.deque(maxlen=50)
        self._check_offline_tts()
        # 预加载常用语句（后台异步）
        self._preload_common_phrases()

    def _check_offline_tts(self):
        self.offline_tts_cmd = None
        if subprocess.call(['which', 'pico2wave'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            self.offline_tts_cmd = 'pico2wave'
            print("[TTS] 离线引擎: pico2wave")
        elif subprocess.call(['which', 'espeak'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            self.offline_tts_cmd = 'espeak'
            print("[TTS] 离线引擎: espeak")
        else:
            print("[TTS] 警告: 无离线 TTS 引擎")

    def _play_audio(self, file_path):
        """通用音频播放：优先 mpg123，回退 aplay"""
        if not os.path.exists(file_path):
            print(f"[播放] 文件不存在: {file_path}")
            return False

        mpg123_path = subprocess.run(['which', 'mpg123'], capture_output=True, text=True).stdout.strip()
        if mpg123_path:
            cmd = [mpg123_path, '-a', 'default', file_path]
            try:
                print(f"[播放] 执行: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                if result.returncode == 0:
                    print("[播放] 成功（mpg123）")
                    return True
                else:
                    print(f"[播放] mpg123 返回码 {result.returncode}")
            except subprocess.TimeoutExpired:
                print("[播放] mpg123 超时")
            except Exception as e:
                print(f"[播放] mpg123 异常: {e}")

        if file_path.endswith('.wav') and subprocess.call(['which', 'aplay'], stdout=subprocess.DEVNULL) == 0:
            cmd = ['aplay', '-f', 'S16_LE', '-r', '16000', '-c', '1', file_path]
            try:
                print(f"[播放] 执行: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                if result.returncode == 0:
                    print("[播放] 成功（aplay）")
                    return True
            except subprocess.TimeoutExpired:
                print("[播放] aplay 超时")
            except Exception as e:
                print(f"[播放] aplay 异常: {e}")

        print("[播放] 所有播放器均失败")
        return False

    def _offline_speak(self, text):
        if self.offline_tts_cmd == 'pico2wave':
            wav_path = os.path.join(TTS_CACHE_DIR, f"offline_{hashlib.md5(text.encode()).hexdigest()}.wav")
            if not os.path.exists(wav_path):
                cmd = ['pico2wave']
                if LANGUAGE == "en":
                    cmd.extend(['-l', 'en-US'])
                else:
                    cmd.extend(['-l', 'zh-CN'])
                cmd.extend(['-w', wav_path, text])
                try:
                    subprocess.run(cmd, timeout=10, check=True)
                    print(f"[TTS离线] 生成 WAV: {wav_path}")
                except Exception as e:
                    print(f"[TTS离线] 生成失败: {e}")
                    return False
            return self._play_audio(wav_path)
        elif self.offline_tts_cmd == 'espeak':
            try:
                subprocess.run(['espeak', text], timeout=10, check=True)
                return True
            except:
                pass
        return False

    def speak(self, text):
        print(f"[机器人 TTS 输出]: {text}")
        self.set_recording(2)

        cache_path = os.path.join(TTS_CACHE_DIR, safe_filename(text))
        if os.path.exists(cache_path):
            if self._play_audio(cache_path):
                self.set_recording(3)
                return
            else:
                os.remove(cache_path)

        for attempt in range(2):
            try:
                resp = requests.post(TTS_URL, json={"model": "tts-1", "input": text, "speed": TTS_SPEED}, timeout=8)
                if resp.status_code == 200 and len(resp.content) > 1024:
                    with open(cache_path, 'wb') as f:
                        f.write(resp.content)
                    if self._play_audio(cache_path):
                        self.set_recording(3)
                        return
                else:
                    print(f"[TTS在线] 状态码 {resp.status_code}, 长度 {len(resp.content)}")
            except requests.exceptions.Timeout:
                print(f"[TTS在线] 超时，尝试 {attempt+1}/2")
                if attempt == 0:
                    time.sleep(1)
                    continue
            except Exception as e:
                print(f"[TTS在线] 异常: {e}")
            break

        self._offline_speak(text)
        self.set_recording(3)

    def set_recording(self, state):
        if state == 1:
            if self.stream is None:
                try:
                    self.stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                                  input=True, frames_per_buffer=CHUNK,
                                                  input_device_index=INPUT_DEVICE_INDEX)
                    time.sleep(0.2)
                    for _ in range(3):
                        self.stream.read(CHUNK, exception_on_overflow=False)
                    print("[录音] 设备打开成功")
                except Exception as e:
                    print(f"[录音] 打开失败: {e}")
                    self.stream = None
        elif state == 0:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                print("[录音] 设备关闭")
        elif state == 2:
            if self.stream and self.stream.is_active():
                self.stream.stop_stream()
        elif state == 3:
            if self.stream and not self.stream.is_active():
                self.stream.start_stream()

    def update_noise_floor(self, frame_energy, is_speech_vad):
        if not is_speech_vad:
            self.energy_history.append(frame_energy)
            if len(self.energy_history) >= 10:
                sorted_energies = sorted(self.energy_history)
                p20_index = max(1, int(len(sorted_energies) * 0.2)) - 1
                self.noise_floor = min(max(sorted_energies[p20_index] * 1.2, 20.0), 8000.0)

    def _apply_agc(self, audio_int16, target_rms=8000):
        if len(audio_int16) == 0: return audio_int16
        rms = np.sqrt(np.mean(audio_int16.astype(np.float32)**2))
        if rms < 1e-6: return audio_int16
        gain = target_rms / rms
        gain = np.clip(gain, 0.3, 3.0)
        audio_float = audio_int16.astype(np.float32) * gain
        audio_float = np.clip(audio_float, -32768, 32767)
        return audio_float.astype(np.int16)

    def _highpass_filter(self, audio_int16, cutoff_hz=80):
        if len(audio_int16) < 2: return audio_int16
        rc = 1.0 / (2 * np.pi * cutoff_hz)
        dt = 1.0 / RATE
        alpha = rc / (rc + dt)
        x = audio_int16.astype(np.float32)
        y = np.zeros_like(x)
        for i in range(1, len(x)):
            y[i] = alpha * (y[i-1] + x[i] - x[i-1])
        return y.astype(np.int16)

    def recognize(self, audio_frames):
        if not audio_frames:
            print("[ASR] 无音频帧")
            return ""
        print(f"[ASR] 收到 {len(audio_frames)} 帧")
        audio_bytes = b''.join(audio_frames)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_int16 = self._highpass_filter(audio_int16)
        audio_int16 = self._apply_agc(audio_int16)

        frame_size = CHUNK * 2
        frames = [audio_int16[i:i+frame_size].tobytes() for i in range(0, len(audio_int16), frame_size)
                  if len(audio_int16[i:i+frame_size]) == frame_size]
        if not frames:
            print("[ASR] 重切分后无有效帧")
            return ""

        temp_path = os.path.join(TTS_CACHE_DIR, f"temp_{int(time.time()*1000)}.wav")
        wf = wave.open(temp_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        print(f"[ASR] 保存临时文件 {temp_path}")

        try:
            with open(temp_path, 'rb') as af:
                resp = requests.post(ASR_URL, files={"audio": ("audio.wav", af, "audio/wav")}, timeout=5)
                print(f"[ASR] HTTP {resp.status_code}")
                if resp.status_code == 200:
                    text = resp.json().get('text', '').strip()
                    text = re.sub(r'<\|[^>]+\|>', '', text)
                    text = re.sub(r'\b(uh|um|ah|eh|er|mm|hmm)\b', '', text, flags=re.IGNORECASE)
                    print(f"[ASR 识别结果]: '{text}'")
                    return text
                else:
                    print(f"[ASR] 错误: {resp.text}")
        except Exception as e:
            print(f"[ASR] 异常: {e}")
        finally:
            try: os.unlink(temp_path)
            except: pass
        return ""

    def record(self):
        if self.stream is None:
            print("[录音] 流未打开")
            return []
        ring_buffer = collections.deque(maxlen=RING_BUFFER_MAXLEN)
        frames = []
        recording = False
        silence_frames = 0
        speech_frames = 0
        total_frames = 0
        max_silence_frames = int(SILENCE_TIMEOUT_MS / FRAME_DURATION_MS)
        max_total_frames = int(MAX_RECORD_DURATION_MS / FRAME_DURATION_MS)
        print("[录音] 开始监听语音...")
        while True:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
            except Exception as e:
                print(f"[录音] 读取错误: {e}")
                break
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
                        print("[录音] 检测到语音，开始录制")
                else:
                    speech_frames = 0
            else:
                frames.append(data)
                if not is_speech:
                    silence_frames += 1
                    if silence_frames >= max_silence_frames:
                        print("[录音] 语音结束（静音超时）")
                        break
                else:
                    silence_frames = 0
            if total_frames >= max_total_frames:
                print("[录音] 达到最大时长")
                break
        print(f"[录音] 录制完成，共 {len(frames)} 帧")
        return frames

    def close(self):
        self.set_recording(0)
        self.audio.terminate()

    def _preload_common_phrases(self):
        """后台预加载常用 TTS 缓存"""
        common_phrases = [
            "Welcome! May I know your name?",
            "Sorry, I didn't catch your name. Could you say it again?",
            "What is your favorite drink?",
            "Sorry, I didn't catch that. What is your favorite drink?",
            "Please have a seat here.",
            "Thank you! Enjoy the party."
        ]
        for phrase in common_phrases:
            threading.Thread(target=self._cache_tts, args=(phrase,), daemon=True).start()

    def _cache_tts(self, text):
        """异步生成 TTS 缓存文件"""
        cache_path = os.path.join(TTS_CACHE_DIR, safe_filename(text))
        if os.path.exists(cache_path):
            return
        try:
            resp = requests.post(TTS_URL, json={"model": "tts-1", "input": text, "speed": TTS_SPEED}, timeout=10)
            if resp.status_code == 200 and len(resp.content) > 1024:
                with open(cache_path, 'wb') as f:
                    f.write(resp.content)
                print(f"[TTS预加载] 已缓存: {text[:30]}...")
        except Exception as e:
            print(f"[TTS预加载] 失败: {e}")

# 为兼容旧接口添加别名
VoiceAssistant.record_utterance = VoiceAssistant.record
VoiceAssistant.recognize_speech = VoiceAssistant.recognize

# ---------- 全局实例 ----------
voice_assistant = VoiceAssistant()

# ---------- 门铃检测部分（测试时可忽略） ----------
if DOORBELL_ENABLED:
    try:
        import librosa
        import sounddevice as sd
        from scipy.spatial.distance import cosine
        DOORBELL_DEPS = True
    except ImportError as e:
        DOORBELL_DEPS = False
        print(f"[门铃] 依赖缺失，自动检测禁用: {e}")

    if DOORBELL_DEPS:
        # ---------- 特征提取 ----------
        def extract_features(y, sr):
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            features = np.concatenate([np.mean(mfcc, axis=1),
                                    np.mean(mfcc_delta, axis=1),
                                    np.mean(mfcc_delta2, axis=1)])
            return features

        def compute_similarity(audio_data, ref_features, sr=RATE):
            if len(audio_data) < 0.5 * sr:
                return 0.0
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            features = extract_features(audio_data, sr)
            sim = 1 - cosine(features, ref_features)
            return sim

        class SoundDeviceDoorbellDetector:
            def __init__(self, template_path, device_index=None,
                         duration=1.5, threshold=0.75, window_size=5, cooldown=0.5):
                self.device_index = device_index
                self.duration = duration
                self.threshold = threshold
                self.window_size = window_size
                self.cooldown = cooldown
                self.running = False
                self.doorbell_detected = threading.Event()
                self.stream_thread = None
                self.ref_features = None

                if not os.path.exists(template_path):
                    raise FileNotFoundError(f"模板不存在: {template_path}")
                y, sr = librosa.load(template_path, sr=RATE)
                max_len = int(RATE * self.duration)
                if len(y) > max_len:
                    y = y[:max_len]
                self.ref_features = extract_features(y, sr)
                print(f"[门铃] 模板加载完成，特征维度 {len(self.ref_features)}")

            def start(self):
                if self.running:
                    return
                self.running = True
                self.doorbell_detected.clear()
                self.stream_thread = threading.Thread(target=self._listen, daemon=True)
                self.stream_thread.start()
                print(f"[门铃] 检测器已启动")

            def stop(self):
                self.running = False
                if self.stream_thread:
                    self.stream_thread.join(timeout=1)
                print("[门铃] 检测器已停止")

            def wait_for_doorbell(self, timeout=None):
                return self.doorbell_detected.wait(timeout)

            def _listen(self):
                sim_buffer = collections.deque(maxlen=self.window_size)
                last_trigger_time = 0
                while self.running:
                    if time.time() - last_trigger_time < self.cooldown:
                        time.sleep(0.2)
                        continue
                    try:
                        audio = sd.rec(int(self.duration * RATE), samplerate=RATE,
                                       channels=1, dtype='float32', device=self.device_index)
                        sd.wait()
                        audio = audio.flatten()
                        rms = np.sqrt(np.mean(audio**2))
                        if rms < 0.005:
                            time.sleep(0.1)
                            continue
                        sim = compute_similarity(audio, self.ref_features, RATE)
                        sim_buffer.append(sim)
                        avg_sim = np.mean(sim_buffer)
                        print(f"[门铃] 相似度={sim:.3f} 平滑={avg_sim:.3f} 阈值={self.threshold}")
                        if avg_sim > self.threshold:
                            print("[门铃] ✅ 检测到匹配的门铃声！")
                            self.doorbell_detected.set()
                            last_trigger_time = time.time()
                    except Exception as e:
                        print(f"[门铃] 检测错误: {e}")
                        time.sleep(0.5)

        _template_path = os.path.join(PROJECT_ROOT, "models", "doorbell_template.wav")
        if os.path.exists(_template_path):
            try:
                doorbell = SoundDeviceDoorbellDetector(
                    template_path=_template_path,
                    device_index=INPUT_DEVICE_INDEX,
                    duration=RECORD_DURATION,
                    threshold=SIMILARITY_THRESHOLD,
                    window_size=SMOOTHING_WINDOW,
                    cooldown=COOLDOWN_SECONDS
                )
                print("[门铃] 全局门铃检测器已创建")
            except Exception as e:
                print(f"[门铃] 创建失败: {e}")
                doorbell = None
        else:
            print(f"[门铃] 模板不存在: {_template_path}，门铃不可用")
            doorbell = None
    else:
        doorbell = None
else:
    doorbell = None
    print("[门铃] 已禁用（DOORBELL_ENABLED=False）")

__all__ = ['voice_assistant', 'doorbell', 'extract_name', 'extract_drink']

# ---------- 全流程测试入口 ----------
def ask_info(prompt, extract_func, retry_prompt, default, max_attempts=2):
    for attempt in range(max_attempts):
        voice_assistant.speak(prompt if attempt == 0 else retry_prompt)
        audio = voice_assistant.record()
        if audio:
            text = voice_assistant.recognize(audio)
            if text:
                info = extract_func(text)
                if info:
                    return info
    return default

def test_full_flow():
    print("\n===== 全流程测试（两位客人）=====")
    input("按 Enter 开始接待第一位客人...")
    
    # 客人1
    name1 = ask_info(
        "Welcome! May I know your name?",
        extract_name,
        "Sorry, I didn't catch your name. Could you say it again?",
        "Friend"
    )
    drink1 = ask_info(
        f"{name1}, what is your favorite drink?",
        extract_drink,
        "Sorry, I didn't catch that. What is your favorite drink?",
        "Water"
    )
    voice_assistant.speak(f"Great, {name1}. Please have a seat.")
    print(f"\n--- 客人1 信息 ---\n姓名: {name1}\n饮品: {drink1}\n")
    
    input("\n按 Enter 开始接待第二位客人...")
    
    # 客人2
    name2 = ask_info(
        "Welcome! May I know your name?",
        extract_name,
        "Sorry, I didn't catch your name. Could you say it again?",
        "Friend"
    )
    drink2 = ask_info(
        f"{name2}, what is your favorite drink?",
        extract_drink,
        "Sorry, I didn't catch that. What is your favorite drink?",
        "Water"
    )
    voice_assistant.speak(f"Great, {name2}. Please have a seat.")
    voice_assistant.speak(f"{name2}, let me introduce {name1}, who likes to drink {drink1}.")
    voice_assistant.speak(f"{name1}, this is {name2}, who likes to drink {drink2}.")
    voice_assistant.speak("Thank you both for coming! Enjoy the party.")
    
    print("\n===== 全流程测试完成 =====")

if __name__ == "__main__":
    # 启用门铃检测
    if doorbell is not None:
        print("等待门铃触发（30秒内）...")
        doorbell.start()
        if doorbell.wait_for_doorbell(timeout=30):
            print("门铃已触发，开始全流程测试")
        else:
            print("等待超时，仍开始测试")
        doorbell.stop()
    else:
        print("门铃检测不可用，直接开始测试")
    
    voice_assistant.set_recording(1)
    test_full_flow()
    voice_assistant.close()