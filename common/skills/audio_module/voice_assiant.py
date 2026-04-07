import time
import threading
from enum import Enum
import pyaudio
import wave
import webrtcvad
import collections
import requests
import io
import pygame
import numpy as np
import noisereduce as nr
import os
import tempfile
import re
import datetime
import hashlib
import subprocess

# ------------------ 门铃检测相关 ------------------
import tensorflow as tf
import tensorflow_hub as hub
import urllib.request


from common.config import ASR_URL, TTS_URL, LLM_BASE_URL, LLM_MODEL

# 录音参数
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

# TTS 缓存目录
TTS_CACHE_DIR = "tts_cache"
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

# 离线 TTS 命令（优先 pico2wave，备选 espeak）
OFFLINE_TTS_CMD = "pico2wave"   # 需要系统安装: sudo apt install libttspico-utils
OFFLINE_TTS_BACKUP = "espeak"   # 需要系统安装: sudo apt install espeak

# ------------------ 门铃检测类 ------------------
class DoorbellDetector:
    def __init__(self, threshold=0.5, chunk_seconds=1.0):
        """
        初始化门铃检测器
        :param threshold: 门铃概率阈值
        :param chunk_seconds: 每次检测的音频长度（秒）
        """
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
        """加载 YAMNet 模型并获取门铃类别索引"""
        print("[门铃] 正在加载 YAMNet 模型...")
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')

        # 下载类别文件
        csv_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
        csv_path = tf.keras.utils.get_file('yamnet_class_map.csv', csv_url)

        class_names = []
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # 跳过标题行
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    class_names.append(parts[2].strip('"'))

        # 查找门铃类别
        for i, name in enumerate(class_names):
            if 'doorbell' in name.lower():
                self.doorbell_index = i
                break

        if self.doorbell_index is None:
            raise ValueError("未找到 'doorbell' 类别，请检查类别文件")
        print(f"[门铃] 门铃类别索引: {self.doorbell_index} ({class_names[self.doorbell_index]})")

    def _detect(self, audio_bytes):
        """检测音频片段是否为门铃"""
        # 转换为 float32 并归一化
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        # 模型需要 shape (samples,)
        scores, embeddings, spectrogram = self.model(audio)
        # scores shape: (frames, 521)
        doorbell_probs = scores[:, self.doorbell_index].numpy()
        avg_prob = np.mean(doorbell_probs)
        return avg_prob > self.threshold

    def start(self):
        """启动门铃监听线程"""
        if self.running:
            return
        self.running = True
        self.doorbell_detected.clear()
        # 初始化音频流
        self.p = pyaudio.PyAudio()
        chunk_size = int(RATE * self.chunk_seconds)
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=chunk_size)
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()
        print("[门铃] 开始监听...")

    def _listen(self):
        """监听线程主循环"""
        while self.running:
            try:
                data = self.stream.read(int(RATE * self.chunk_seconds), exception_on_overflow=False)
                if self._detect(data):
                    print("[门铃] 检测到门铃！")
                    self.doorbell_detected.set()
                    # 为避免重复触发，等待一段时间
                    time.sleep(2)
            except Exception as e:
                print(f"[门铃] 检测异常: {e}")
                time.sleep(0.5)

    def stop(self):
        """停止监听并释放资源"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        print("[门铃] 监听已停止")

    def wait_for_doorbell(self, timeout=None):
        """等待门铃，返回是否检测到（可设置超时）"""
        return self.doorbell_detected.wait(timeout)

# ------------------ 名字提取函数 ------------------
def extract_name(text):
    """从自然语句中提取名字（规则）"""
    if not text:
        return ""

    text = text.strip()

    patterns = [
        r'(?:我叫|我是|名字是|姓名是|本人叫|本人是)\s*([^\s，。、]+)',
        r'叫\s*([^\s，。、]+)',
        r'^([^\s，。、]{2,4})$',
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1)

    english_patterns = [
        r"(?i)\b(?:my name is|i am|i'm|im|this is)\s*([A-Za-z][A-Za-z'\-]{0,31})\b",
        r"(?i)\bname\s*(?:is)?\s*([A-Za-z][A-Za-z'\-]{0,31})\b",
        r"(?i)^([A-Za-z][A-Za-z'\-]{1,31})$",
    ]
    for pattern in english_patterns:
        m = re.search(pattern, text)
        if m:
            name = m.group(1)
            return _normalize_english_name(name)

    compact_text = re.sub(r"[^A-Za-z]", "", text)
    compact_patterns = [
        r"(?i)(?:mynameis|iam|im|thisis)([A-Za-z]{2,32})$",
        r"(?i)^([A-Za-z]{2,32})$",
    ]
    for pattern in compact_patterns:
        m = re.search(pattern, compact_text)
        if m:
            name = m.group(1)
            return _normalize_english_name(name)

    words = re.findall(r'[\u4e00-\u9fff]+', text)
    return words[-1] if words else text


def _normalize_english_name(name):
    name = re.sub(r"[^A-Za-z'\-]", "", name).strip("-'")
    if not name:
        return ""
    return "-".join(part.capitalize() for part in name.split("-"))

# ------------------ 语音助手类 ------------------
class VoiceAssistant:
    def __init__(self, use_rnnoise=True):
        self.use_noise_suppression = use_rnnoise
        self.vad = webrtcvad.Vad(3)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.history = []
        self._init_system_prompt()
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        self.noise_floor = 500.0
        self.energy_history = collections.deque(maxlen=50)
        self.is_speech_threshold = 1.2
        self.COMMON_DRINKS = ["可乐", "雪碧", "芬达", "美年达", "七喜", "果汁", "橙汁", "苹果汁",
                              "牛奶", "酸奶", "水", "矿泉水", "茶", "红茶", "绿茶", "乌龙茶",
                              "咖啡", "拿铁", "卡布奇诺", "啤酒", "红酒"]
        if self.use_noise_suppression:
            print("[信息] noisereduce降噪已启用")
        else:
            print("[信息] 降噪已禁用")

        # 检测离线 TTS 可用性
        self.offline_tts_available = self._check_offline_tts()

    def _check_offline_tts(self):
        """检查 pico2wave 是否可用"""
        try:
            subprocess.run([OFFLINE_TTS_CMD, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1)
            print("[信息] 离线 TTS (pico2wave) 可用")
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            print("[警告] 未找到 pico2wave，将使用 espeak 作为备选（中文效果差）")
            try:
                subprocess.run([OFFLINE_TTS_BACKUP, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1)
                print("[信息] 离线 TTS (espeak) 可用")
                return True
            except:
                print("[警告] 离线 TTS 不可用，将仅使用在线 TTS")
                return False

    def _offline_tts(self, text):
        """使用离线 TTS 播放文本，返回是否成功"""
        if not self.offline_tts_available:
            return False

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            # 尝试 pico2wave 或 espeak
            if subprocess.run([OFFLINE_TTS_CMD, "-w", temp_path, text], check=True, timeout=5).returncode == 0:
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                time.sleep(0.3)
                os.unlink(temp_path)
                return True
            else:
                # 尝试 espeak
                subprocess.run([OFFLINE_TTS_BACKUP, "-w", temp_path, text], check=True, timeout=5)
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                time.sleep(0.3)
                os.unlink(temp_path)
                return True
        except Exception as e:
            print(f"[离线TTS失败] {e}")
            return False

    def _get_cache_path(self, text):
        """根据文本内容生成缓存文件路径"""
        cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
        return os.path.join(TTS_CACHE_DIR, f"{cache_key}.mp3")

    def preload_tts(self, text):
        """异步预加载文本音频（不播放）"""
        cache_path = self._get_cache_path(text)
        if os.path.exists(cache_path):
            return
        try:
            payload = {"model": "tts-1", "input": text, "voice": "alloy", "speed": 1.0}
            resp = requests.post(TTS_URL, json=payload, timeout=10)
            if resp.status_code == 200:
                with open(cache_path, 'wb') as f:
                    f.write(resp.content)
                print(f"[预加载] 已缓存: {text}")
        except Exception as e:
            print(f"[预加载失败] {e}")

    def speak(self, text, max_retries=1, timeout=5):
        """
        播放语音，优先使用缓存，其次在线 TTS，最后离线备选
        :param text: 要播放的文本
        :param max_retries: 在线 TTS 重试次数
        :param timeout: 在线 TTS 单次超时（秒）
        """
        print(f"[机器人]: {text}")
        self.pause_stream()

        # 1. 检查缓存
        cache_path = self._get_cache_path(text)
        if os.path.exists(cache_path):
            try:
                pygame.mixer.music.load(cache_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                time.sleep(0.3)
                self.resume_stream()
                return
            except Exception as e:
                print(f"[缓存播放失败] {e}")

        # 2. 在线 TTS
        success = False
        for attempt in range(max_retries):
            try:
                payload = {"model": "tts-1", "input": text, "voice": "alloy", "speed": 1.0}
                resp = requests.post(TTS_URL, json=payload, timeout=timeout)
                if resp.status_code == 200:
                    # 保存到缓存
                    with open(cache_path, 'wb') as f:
                        f.write(resp.content)
                    # 播放
                    pygame.mixer.music.load(io.BytesIO(resp.content))
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    time.sleep(0.3)
                    success = True
                    break
                else:
                    print(f"[TTS错误] 状态码 {resp.status_code}，尝试 {attempt+1}/{max_retries}")
            except Exception as e:
                print(f"[TTS异常] {e}，尝试 {attempt+1}/{max_retries}")
            time.sleep(1)

        # 3. 离线 TTS 备选
        if not success:
            if self._offline_tts(text):
                success = True
            else:
                print(f"[TTS失败] 无法播放语音，文本内容: {text}")

        self.resume_stream()

    def update_noise_floor(self, frame_energy, is_speech_vad):
        if not is_speech_vad:
            self.energy_history.append(frame_energy)
            if len(self.energy_history) >= 10:
                sorted_energies = sorted(self.energy_history)
                p20_index = max(1, int(len(sorted_energies) * 0.2)) - 1
                noise_est = sorted_energies[p20_index]
                self.noise_floor = min(max(noise_est * 1.2, 20.0), 8000.0)

    def calibrate_noise(self, duration_ms=1000):
        if self.stream is None:
            print("[警告] 音频流未打开，跳过噪声校准")
            return
        print("[调试] 正在校准噪声基底，请保持安静...")
        frames_needed = int(duration_ms / FRAME_DURATION_MS)
        energies = []
        for _ in range(frames_needed):
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            audio_array = np.frombuffer(data, dtype=np.int16)
            energy = np.sqrt(np.mean(audio_array.astype(float)**2))
            energies.append(energy)
        median_energy = float(np.median(energies))
        self.noise_floor = min(max(median_energy * 1.2, 20.0), 8000.0)
        self.energy_history.extend(energies)
        print(f"[调试] 校准完成，噪声基底 = {self.noise_floor:.1f}")

    def denoise_with_noisereduce(self, audio_frames):
        if not audio_frames:
            return audio_frames
        try:
            audio_bytes = b''.join(audio_frames)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            noise_sample_len = int(0.5 * RATE)
            noise_sample = audio_float32[:min(noise_sample_len, len(audio_float32))]
            reduced = nr.reduce_noise(
                y=audio_float32,
                sr=RATE,
                y_noise=noise_sample if len(noise_sample) > 0 else None,
                prop_decrease=1.0,
                time_constant_s=2.0,
                freq_mask_smooth_hz=500,
                time_mask_smooth_ms=50,
                thresh_n_mult_nonstationary=2,
                sigmoid_slope_nonstationary=10,
                n_std_thresh_stationary=1.5,
                stationary=False,
                tmp_folder=None,
                chunk_size=600000,
                padding=30000,
                n_fft=1024,
                win_length=1024,
                hop_length=256,
            )
            reduced_int16 = (reduced * 32768).astype(np.int16)
            frame_bytes = reduced_int16.tobytes()
            frame_size = CHUNK * 2
            denoised_frames = [
                frame_bytes[i:i+frame_size]
                for i in range(0, len(frame_bytes), frame_size)
                if len(frame_bytes[i:i+frame_size]) == frame_size
            ]
            if len(denoised_frames) < len(audio_frames):
                denoised_frames.extend([denoised_frames[-1]] * (len(audio_frames) - len(denoised_frames)))
            elif len(denoised_frames) > len(audio_frames):
                denoised_frames = denoised_frames[:len(audio_frames)]
            return denoised_frames
        except Exception as e:
            print(f"[noisereduce降噪异常] {e}，返回原始音频")
            return audio_frames

    def clean_asr_text(self, text):
        if not text:
            return ""
        cleaned = re.sub(r'<\|[^>]+\|>', '', text)
        cleaned = re.sub(r'[嗯啊哦呃诶]+', '', cleaned)
        cleaned = re.sub(r'[，。！？、\s]+', '', cleaned)
        cleaned = re.sub(r'\s+', '', cleaned).strip()
        return cleaned

    def close(self):
        self.stop_stream()
        self.audio.terminate()
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()

    def _init_system_prompt(self):
        self.system_prompt = (
            "你是一个派对接待机器人，名叫RoboHost。你的任务是热情迎接客人，询问他们的名字和最喜欢的饮料，"
            "引导他们就座，并在所有客人到达后介绍他们。说话要礼貌、友好，每次回答不超过两句话。"
        )
        self.history = [{"role": "system", "content": self.system_prompt}]

    def start_stream(self):
        if self.stream is None:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            time.sleep(0.3)
            for _ in range(5):
                try:
                    self.stream.read(CHUNK, exception_on_overflow=False)
                except:
                    pass

    def stop_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def pause_stream(self):
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()

    def resume_stream(self):
        if self.stream and not self.stream.is_active():
            self.stream.start_stream()
            for _ in range(5):
                try:
                    self.stream.read(CHUNK, exception_on_overflow=False)
                except:
                    pass

    def record_utterance(self):
        ring_buffer = collections.deque(maxlen=RING_BUFFER_MAXLEN)
        frames = []
        is_recording = False
        silence_frames = 0
        speech_frames = 0
        max_silence_frames = int(SILENCE_TIMEOUT_MS / FRAME_DURATION_MS)
        max_total_frames = int(MAX_RECORD_DURATION_MS / FRAME_DURATION_MS)
        total_frames = 0
        debug_interval = 50
        low_energy_frames = 0

        while True:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            audio_array = np.frombuffer(data, dtype=np.int16)
            frame_energy = np.sqrt(np.mean(audio_array.astype(float)**2))
            is_speech_vad = self.vad.is_speech(data, RATE)
            self.update_noise_floor(frame_energy, is_speech_vad)
            energy_threshold = self.noise_floor * self.is_speech_threshold
            is_speech_energy = frame_energy > energy_threshold
            is_speech = is_speech_vad and is_speech_energy
            total_frames += 1

            if total_frames % debug_interval == 0:
                print(f"[调试] 帧:{total_frames}, 能量:{frame_energy:.1f}, 阈值:{energy_threshold:.1f}, VAD:{is_speech_vad}, 综合语音:{is_speech}")

            if not is_recording:
                ring_buffer.append(data)
                if is_speech:
                    speech_frames += 1
                    if speech_frames >= SPEECH_START_FRAMES:
                        frames.extend(ring_buffer)
                        is_recording = True
                        print(f"[调试] 开始录音... (能量: {frame_energy:.1f} > {energy_threshold:.1f})")
                else:
                    speech_frames = 0
            else:
                frames.append(data)
                if is_speech_energy:
                    low_energy_frames = 0
                else:
                    low_energy_frames += 1
                    if low_energy_frames >= MAX_LOW_ENERGY_FRAMES:
                        print("[调试] 录音结束（低能量持续）")
                        break
                if is_speech:
                    silence_frames = 0
                else:
                    silence_frames += 1
                    if silence_frames >= max_silence_frames:
                        print(f"[调试] 录音结束（静音超时），静音帧数:{silence_frames}")
                        break
            if total_frames >= max_total_frames:
                print(f"[调试] 录音结束（达到最大时长 {MAX_RECORD_DURATION_MS/1000}秒）")
                break
        return frames

    def recognize_speech(self, audio_frames):
        if self.use_noise_suppression:
            denoised_frames = self.denoise_with_noisereduce(audio_frames)
        else:
            denoised_frames = audio_frames

        debug_dir = "debug_audio"
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        raw_path = os.path.join(debug_dir, f"raw_{timestamp}.wav")
        with wave.open(raw_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(audio_frames))
        denoised_path = os.path.join(debug_dir, f"denoised_{timestamp}.wav")
        with wave.open(denoised_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(denoised_frames))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wf = wave.open(f, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(denoised_frames))
            wf.close()
            temp_path = f.name

        try:
            with open(temp_path, 'rb') as audio_file:
                files = {"audio": ("audio.wav", audio_file, "audio/wav")}
                resp = requests.post(ASR_URL, files=files, timeout=5)
                if resp.status_code == 200:
                    text = resp.json().get('text', '').strip()
                    text = self.clean_asr_text(text)
                    return text
                else:
                    print(f"[ASR错误] {resp.status_code}")
                    return ""
        except Exception as e:
            print(f"[ASR异常] {e}")
            return ""
        finally:
            os.unlink(temp_path)

    def ask_llm(self, user_input):
        url = f"{LLM_BASE_URL}/chat/completions"
        self.history.append({"role": "user", "content": user_input})
        data = {
            "model": LLM_MODEL,
            "messages": self.history,
            "stream": False,
            "temperature": 0.7
        }
        try:
            resp = requests.post(url, json=data, timeout=30)
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LLM错误] {e}")
            reply = "抱歉，我暂时无法回答。"
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def ask_llm_single(self, prompt, system_prompt=None):
        """调用 LLM 进行单次对话，不保存历史"""
        url = f"{LLM_BASE_URL}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,
            "temperature": 0.3
        }
        try:
            resp = requests.post(url, json=data, timeout=10)
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"].strip()
            return reply
        except Exception as e:
            print(f"[LLM单次调用错误] {e}")
            return None
