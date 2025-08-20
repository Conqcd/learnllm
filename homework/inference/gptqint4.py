from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

recipe = [
    GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
]
oneshot(
    model="Qwen/Qwen2-7B-Instruct",
    dataset="open_platypus",
    recipe=recipe,
    output_dir="Qwen/Qwen2-7B-Instruct-INT8",
    max_seq_length=2048,
    num_calibration_samples=512,
)