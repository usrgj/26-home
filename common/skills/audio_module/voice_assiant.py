#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整语音助手模块 - 稳定版（改进 TTS 缓存机制）
- 使用基于内容的可读文件名缓存（例如 "Welcome!.mp3"）
- 支持预加载常用语句
- 门铃检测可选，依赖缺失时 doorbell 为 None
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
DOORBELL_ENABLED = True
INPUT_DEVICE_INDEX = None
RECORD_DURATION = 1.5
SIMILARITY_THRESHOLD = 0.75
SMOOTHING_WINDOW = 5
COOLDOWN_SECONDS = 0.5
TTS_SPEED = 1.0

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

# ---------- 辅助函数：生成安全文件名 ----------
def safe_filename(text: str, max_len=100) -> str:
    """将文本转换为安全、可读的文件名（例如 "Welcome!.mp3"）"""
    # 替换非法字符（Windows/Linux 通用）
    illegal_chars = r'[\\/*?:"<>|]'
    safe = re.sub(illegal_chars, '_', text)
    # 去除首尾空格和点号
    safe = safe.strip('. ')
    # 限制长度
    if len(safe) > max_len:
        safe = safe[:max_len]
    return safe + ".mp3"

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

# ---------- 名字和饮料提取 ----------
def extract_name_en(text):
    if not text:
        return None
    lower_text = text.lower()
    cleaned = text
    prefixes = [
        "my name is ", "i am ", "i'm ", "name is ", "this is ", "call me ",
        "i m ", "im ", "un ", "um ", "uh ", "er ", "and ", "so "
    ]
    for prefix in prefixes:
        if lower_text.startswith(prefix):
            cleaned = text[len(prefix):]
            break
    if cleaned.lower().startswith("un "):
        cleaned = cleaned[3:]
    doc = nlp(cleaned)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            if len(name) >= 2 and name[0].isupper():
                return name
    patterns = [r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"]
    for p in patterns:
        m = re.search(p, cleaned)
        if m:
            name = m.group(1).strip()
            if len(name) >= 2 and name[0].isupper():
                return name
    words = re.findall(r'[A-Z][a-z]+', cleaned)
    if words:
        if len(words) >= 2 and len(words[0]) > 1 and len(words[1]) > 1:
            return f"{words[0]} {words[1]}"
        return words[0]
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
    def _get_cache_path(self, text):
        # 注意：这里必须导入 safe_filename，或者定义相同函数
        return os.path.join(TTS_CACHE_DIR, safe_filename(text))

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
        """通用音频播放：优先 mpg123（支持 MP3 和 WAV），回退 aplay"""
        if not os.path.exists(file_path):
            print(f"[播放] 文件不存在: {file_path}")
            return False

        # 方法1: mpg123（支持 MP3 和 WAV）
        mpg123_path = subprocess.run(['which', 'mpg123'], capture_output=True, text=True).stdout.strip()
        if mpg123_path:
            cmd = [mpg123_path, '-a', 'default', file_path]
            try:
                print(f"[播放] 执行: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)  # 增加超时到15秒
                if result.returncode == 0:
                    print("[播放] 成功（mpg123）")
                    return True
                else:
                    print(f"[播放] mpg123 返回码 {result.returncode}")
            except subprocess.TimeoutExpired:
                print("[播放] mpg123 超时")
            except Exception as e:
                print(f"[播放] mpg123 异常: {e}")

        # 方法2: aplay（仅 WAV，作为后备）
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
            # 使用统一播放函数
            return self._play_audio(wav_path)
        elif self.offline_tts_cmd == 'espeak':
            try:
                subprocess.run(['espeak', text], timeout=10, check=True)
                return True
            except:
                pass
        return False

    def speak(self, text):
        print(f"[机器人]: {text}")
        self.set_recording(2)

        # 使用可读文件名缓存
        cache_mp3 = os.path.join(TTS_CACHE_DIR, safe_filename(text))
        if os.path.exists(cache_mp3):
            if self._play_audio(cache_mp3):
                self.set_recording(3)
                return
            else:
                os.remove(cache_mp3)  # 损坏则删除

        # 在线 TTS（增加超时和重试）
        for attempt in range(2):  # 重试一次
            try:
                resp = requests.post(TTS_URL, json={"model": "tts-1", "input": text, "speed": TTS_SPEED}, timeout=8)
                if resp.status_code == 200 and len(resp.content) > 1024:
                    with open(cache_mp3, 'wb') as f:
                        f.write(resp.content)
                    if self._play_audio(cache_mp3):
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
            break  # 非超时错误不重试

        # 离线 TTS 降级
        self._offline_speak(text)
        self.set_recording(3)

    def _preload_tts(self, text):
        """预生成单条 TTS 音频缓存（不播放）"""
        cache_path = self._get_cache_path(text)
        if os.path.exists(cache_path):
            return
        try:
            resp = requests.post(TTS_URL, json={"model": "tts-1", "input": text, "speed": TTS_SPEED}, timeout=5)
            if resp.status_code == 200 and len(resp.content) > 1024:
                with open(cache_path, 'wb') as f:
                    f.write(resp.content)
                print(f"[TTS预加载] 已缓存: {text[:30]}...")
        except Exception as e:
            print(f"[TTS预加载] 失败: {e}")

    def _preload_common_phrases(self):
        """异步预加载常用语句"""
        if LANGUAGE == "en":
            phrases = [
                "Welcome! May I know your name?",
                "Sorry, I didn't catch your name. Could you say it again?",
                "What is your favorite drink?",
                "Sorry, I didn't catch that. What is your favorite drink?",
                "Please have a seat here.",
                "Thank you! Enjoy the party."
            ]
        else:
            phrases = [
                "你好，欢迎来到派对！请问你叫什么名字？",
                "抱歉，我没听清你的名字，能再说一遍吗？",
                "你最喜欢的饮料是什么？",
                "抱歉，我没听清你喜欢的饮料，能再说一次吗？",
                "请坐在这里。",
                "谢谢！祝您玩得开心。"
            ]
        for phrase in phrases:
            threading.Thread(target=self._preload_tts, args=(phrase,), daemon=True).start()

    def set_recording(self, state):
        if state == 1:
            if self.stream is None:
                try:
                    self.stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                                  input=True, frames_per_buffer=CHUNK,
                                                  input_device_index=None)
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
                    print(f"[ASR] 识别文本: '{text}'")
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

# 为了兼容旧接口
VoiceAssistant.record_utterance = VoiceAssistant.record
VoiceAssistant.recognize_speech = VoiceAssistant.recognize

# ---------- 全局实例 ----------
voice_assistant = VoiceAssistant()

# ---------- 门铃检测部分（保持不变） ----------
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
    print("[门铃] 已禁用")

__all__ = ['voice_assistant', 'doorbell', 'extract_name', 'extract_drink']

# ---------- 独立测试入口 ----------
if __name__ == "__main__":
    print("独立测试模式：5秒内播放门铃将触发检测")
    if doorbell:
        doorbell.start()
        print("等待门铃... (10秒)")
        if doorbell.wait_for_doorbell(10):
            print("检测到门铃！")
        else:
            print("超时，未检测到")
        doorbell.stop()
    else:
        print("门铃不可用，测试语音合成:")
        voice_assistant.speak("Hello, this is a test.")
