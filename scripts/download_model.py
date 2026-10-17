"""下载 BGE 中文向量模型到本地 models/ 目录
国内环境从 ModelScope 下载（比 HuggingFace 快且稳定），下载后离线可用。

用法: .venv\\Scripts\\python scripts\\download_model.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "bge-small-zh-v1.5")


def main() -> None:
    if os.path.isdir(LOCAL_DIR) and os.listdir(LOCAL_DIR):
        print(f"✅ 模型已存在: {LOCAL_DIR}")
        return

    print(f"从 ModelScope 下载 {MODEL_NAME} ...（约 100MB，国内较快）")
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("❌ 未安装 modelscope，先执行: pip install modelscope")
        sys.exit(1)

    snapshot_download(MODEL_NAME, local_dir=LOCAL_DIR)
    print(f"✅ 下载完成: {LOCAL_DIR}")


if __name__ == "__main__":
    main()
