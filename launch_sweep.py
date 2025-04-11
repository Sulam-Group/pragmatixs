import os
import subprocess
import tempfile
import zipfile
import argparse
from pathlib import Path
import shutil


def get_git_commit():
    """Get current Git commit hash."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        return commit
    except Exception as e:
        print(f"Warning: Unable to get git commit. {e}")
        return "unknown"


def zip_code_with_excludes(zip_path, exclude_dirs=None, exclude_exts=None):
    """
    Create a zip archive of the current directory, excluding certain folders and file types.

    Args:
        zip_path (str): Output path for the zip file.
        exclude_dirs (set): Folder names to exclude (e.g., {".git", "wandb", "logs"}).
        exclude_exts (set): File extensions to exclude (e.g., {".log", ".pyc"}).
    """
    exclude_dirs = exclude_dirs or {".venv", "wandb", "weights", "logs", "__pycache__", "data", "data", "embed_cache"}
    exclude_exts = exclude_exts or {".log", ".pyc", ".tmp", ".git", "nohup.out", '.gitignore'}

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in Path(".").rglob("*"):
            if file.is_file():
                root_dir = Path('.').resolve()
                try:
                    relative_path = file.resolve().relative_to(root_dir.resolve())
                except ValueError:
                    print(f"Skipping file not under root: {file}")
                    continue
                # Skip files in excluded directories
                if any(part in exclude_dirs for part in relative_path.parts):
                    continue
                # Skip files with excluded extensions
                if file.suffix in exclude_exts:
                    continue
                zipf.write(file, relative_path)


def run_agent_on_gpu(sweep_id, zip_path, commit, gpu_id):
    """Run wandb agent on a specific GPU from zipped code snapshot."""
    temp_dir = tempfile.mkdtemp()
    print(f"[GPU {gpu_id}] → Using temp dir: {temp_dir}")

    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(temp_dir)

    # Save commit for logging
    with open(Path(temp_dir) / "GIT_COMMIT.txt", "w") as f:
        f.write(commit)

    # Launch wandb agent with GPU binding
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] → Launching agent")
    log_file = open(f"gpu{gpu_id}_agent.log", "w")
    subprocess.Popen(
        ["wandb", "agent", sweep_id],
        cwd=temp_dir,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Launch W&B sweep on 3 GPUs in parallel."
    )
    parser.add_argument(
        "--sweep_id", required=True, help="W&B sweep ID (e.g., user/project/sweepid)"
    )
    parser.add_argument(
        "--zip_path",
        default="sweep_code_snapshot.zip",
        help="Path for the zipped snapshot",
    )
    parser.add_argument(
        "--gpus", nargs="+", type=int, default=[0, 1, 2], help="List of GPU IDs to use"
    )
    args = parser.parse_args()

    print("🔍 Getting git commit...")
    commit = get_git_commit()
    print(f"✅ Git commit: {commit}")

    print("📦 Creating code snapshot...")
    zip_code_with_excludes(args.zip_path)
    print(f"✅ Code archived at {args.zip_path}")

    print("🚀 Launching sweep agents on GPUs:", args.gpus)
    for gpu in args.gpus:
        run_agent_on_gpu(args.sweep_id, args.zip_path, commit, gpu)

    print("✅ All agents started in background. Check W&B for progress.")


if __name__ == "__main__":
    main()
