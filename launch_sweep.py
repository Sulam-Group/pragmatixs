import os
import subprocess
import argparse
def run_agent_on_gpu(sweep_id, gpu_id):
    """Run wandb agent on a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_file = open(f"gpu{gpu_id}_agent.log", "w")
    print(f"[GPU {gpu_id}] → Launching wandb agent...")
    subprocess.Popen(
        ["wandb", "agent", sweep_id],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
def main():
    parser = argparse.ArgumentParser(description="Launch W&B sweep on multiple GPUs.")
    parser.add_argument(
        "--sweep_id", required=True, help="W&B sweep ID (e.g., user/project/sweepid)"
    )
    parser.add_argument(
        "--gpus", nargs="+", type=int, default=[0, 1, 2], help="List of GPU IDs to use"
    )
    args = parser.parse_args()

    print("🚀 Launching sweep agents on GPUs:", args.gpus)
    for gpu in args.gpus:
        run_agent_on_gpu(args.sweep_id, gpu)

    print("✅ All agents started in background. Check W&B for progress.")


if __name__ == "__main__":
    main()
