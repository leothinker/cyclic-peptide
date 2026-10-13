"""Background-download PatSnap/Hiro-OCSR checkpoint to ./.hiro-model/."""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path(".hiro-model")
target.mkdir(exist_ok=True)
print(f"snapshot_download -> {target.resolve()} (this is large; 16GB Qwen3-VL-8B)")
p = snapshot_download(
    repo_id="PatSnap/Hiro-OCSR",
    local_dir=str(target),
    allow_patterns=[
        "*.json", "*.txt", "*.jinja", "*.md",
        "*.safetensors", "*.bin",
    ],
    max_workers=4,
)
print("OK:", p)
