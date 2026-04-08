# Intro
这个代码包含除虚拟环境外的所有云端代码。配置完虚拟环境之后，直接用下面的终端指令就可以运行

# CPU 

python3 -m inference.inference_op_sr \
  --model_type htdemucs \
  --config_path configs/config_htdemucs_6stems.yaml \
  --input_folder data/inputs/long_test \
  --store_dir data/outputs \
  --use_onnx \
  --onnx_model_path models/htdemucs_6s.onnx \
  --output_sr 48000 \
  --force_cpu

# GPU
python3 -m inference.inference_op_sr \
  --model_type htdemucs \
  --config_path configs/config_htdemucs_6stems.yaml \
  --input_folder data/inputs/long_test \
  --store_dir data/outputs_1 \
  --use_onnx \
  --onnx_model_path models/htdemucs_6s.onnx \
  --output_sr 48000


short_test
# 联网

uvicorn server.app:app --host 0.0.0.0 --port 8000

