from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openai/gpt-oss-20b",
    local_dir="models/openai-gpt-oss-20b",
    local_dir_use_symlinks=False,
)
