#!/usr/bin/env python3
"""
Download a local LLM from HuggingFace for imputation tasks.

Recommended models for local execution:
- microsoft/phi-2: 2.7B params, good for smaller systems
- TinyLlama/TinyLlama-1.1B-Chat-v1.0: 1.1B params, very lightweight
- google/gemma-2b: 2B params, good quality
- meta-llama/Llama-3.2-1B: 1B params, latest from Meta (requires approval)
"""

import argparse
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default model: phi-2 is a good balance of quality and size
DEFAULT_MODEL = "microsoft/phi-2"
DEFAULT_DIR = "models/phi-2"

def download_model(repo_id: str, local_dir: str):
    """Download a model from HuggingFace Hub."""
    local_path = Path(local_dir)

    if local_path.exists():
        logger.warning(f"Directory {local_dir} already exists. Checking if download is needed...")

    logger.info(f"Downloading model: {repo_id}")
    logger.info(f"Target directory: {local_dir}")
    logger.info("This may take a while depending on model size and internet speed...")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        logger.info(f"Successfully downloaded {repo_id} to {local_dir}")
        logger.info(f"To use this model, run: python -m src.run --llm openai-oss --data <your_data.csv> --target <target_col>")
        logger.info(f"Or set the model path in your config to: {local_path.resolve()}")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        logger.error("Make sure you have access to the model and are logged in with `huggingface-cli login` if needed")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a local LLM from HuggingFace")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model repo ID (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--dir",
        default=DEFAULT_DIR,
        help=f"Local directory to save model (default: {DEFAULT_DIR})"
    )

    args = parser.parse_args()
    download_model(args.model, args.dir)
