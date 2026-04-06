import subprocess
import time
import os
import signal
import sys
import threading

class LlamaServer:
    def __init__(self, model_path, host="127.0.0.1", port=8000,
                 context_size=2048, batch_size=512, parallel=1,
                 verbose=False, chat_template=None):
        """
        初始化 llama.cpp 服务器管理器。
        :param model_path: GGUF 模型文件的路径
        :param host: 绑定的主机地址
        :param port: 绑定的端口
        :param context_size: 上下文大小 (-c)
        :param batch_size: 批处理大小 (-b)
        :param parallel: 并行序列数 (-np)
        :param verbose: 是否输出详细日志 (--verbose)
        :param chat_template: 自定义 chat template 文件路径 (--chat-template)
        """
        self.model_path = model_path
        self.host = host
        self.port = port
        self.context_size = context_size
        self.batch_size = batch_size
        self.parallel = parallel
        self.verbose = verbose
        self.chat_template = chat_template
        self.process = None

    def start(self, server_executable="./bin/llama-server"):
        """
        启动服务器进程。
        :param server_executable: llama-server 可执行文件的路径
        """
        if self.process is not None:
            print("服务器已在运行中")
            return

        cmd = [
            server_executable,
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.context_size),
            "-b", str(self.batch_size),
            "-np", str(self.parallel)
        ]
        if self.verbose:
            cmd.append("--verbose")
        if self.chat_template:
            cmd.extend(["--chat-template", self.chat_template])

        print(f"启动命令: {' '.join(cmd)}")
        try:
            # 使用 Popen 启动子进程，并将输出重定向到管道（或直接打印）
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            def log_output():
                for line in self.process.stdout:
                    print(f"[llama-server] {line}", end="")
            threading.Thread(target=log_output, daemon=True).start()

            # 等待一小会儿确认进程是否存活
            time.sleep(2)
            if self.process.poll() is not None:
                print("服务器启动失败，请检查错误信息。如果启用了 verbose，上面的输出可能包含详细原因。")
                self.process = None
            else:
                print(f"服务器已启动，监听 {self.host}:{self.port}")
        except Exception as e:
            print(f"启动服务器时出错: {e}")
            self.process = None

    def stop(self):
        """停止服务器进程"""
        if self.process is None:
            print("服务器未运行")
            return
        print("正在停止服务器...")
        self.process.terminate()   # 发送 SIGTERM
        try:
            self.process.wait(timeout=5)  # 等待最多5秒
        except subprocess.TimeoutExpired:
            print("服务器未响应，强制终止")
            self.process.kill()
            self.process.wait()
        self.process = None
        print("服务器已停止")

    def is_running(self):
        """检查服务器是否仍在运行"""
        if self.process is None:
            return False
        return self.process.poll() is None

# 使用示例
if __name__ == "__main__":
    # 根据你的实际路径调整
    MODEL_PATH = "/home/luo/桌面/api/llama.cpp/models/qwen1_5-1_8b-chat-q8_0.gguf"
    EXECUTABLE = "/home/luo/桌面/api/llama.cpp/build/bin/llama-server"

    # 如果需要调试 system 提示词，可设置 verbose=True 并指定 chat_template
    # 首先创建一个 chatml.jinja 文件，内容见下方注释
    server = LlamaServer(
        model_path=MODEL_PATH,
        host="127.0.0.1",
        port=8000,
        context_size=2048,
        batch_size=512,
        parallel=1,
        verbose=False,                # 开启详细日志
        # chat_template="/home/luo/桌面/api/llama.cpp/chatml.jinja"  # 如果需要自定义模板，取消注释
    )

    try:
        server.start(server_executable=EXECUTABLE)
        print("按 Ctrl+C 停止服务器...")
        while server.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到中断信号")
    finally:
        server.stop()
