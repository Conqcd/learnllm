#模型下载
from modelscope import snapshot_download
cache_dir = '/data2/anti_fraud/models/modelscope/hub'
model_dir = snapshot_download('Qwen/Qwen2-7B-Instruct', cache_dir=cache_dir, revision='master')
# model_dir = snapshot_download('Qwen/Qwen2-0.5B-Instruct', cache_dir=cache_dir, revision='master')
# model_dir = snapshot_download('Qwen/Qwen2-1.5B-Instruct', cache_dir=cache_dir, revision='master')
# model_dir = snapshot_download('Qwen/Qwen2-1.5B', cache_dir=cache_dir, revision='master')
model_dir