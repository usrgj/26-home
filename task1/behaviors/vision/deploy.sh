# Explicitly disable V1 and force-set the backend to V0
# 需要在conda的 vllm_deploy 环境中运行
export VLLM_USE_V1=0

python3 -m swift.cli.deploy \
  --model /home/blinx/api_servers/models/Qwen3.5-4B-FP8 \
  --infer_backend vllm \
  --max_new_tokens 4096 \
  --api_key retoo \
  --served_model_name Qwen3.5-4B \
  --host 0.0.0.0 \
  --port 8004 \
  --vllm_gpu_memory_utilization 0.7 \
  --vllm_tensor_parallel_size 1 \
  --vllm_max_model_len 32768 \
  --vllm_enforce_eager True \
  --verbose