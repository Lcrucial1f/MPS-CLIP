import argparse
import subprocess
import os
import sys

# Run segmentation.py on several GPUs, one process per GPU.
# Arguments other than --gpus are passed through to segmentation.py, e.g.
#   python launcher.py --gpus 0,1,2,3 --json_path ... --image_root ... --out_root ...
parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids, e.g. 0,1,2,3")
args, seg_args = parser.parse_known_args()

gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
world_size = len(gpus)
script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "segmentation.py")

procs = []
for rank, gpu in enumerate(gpus):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    cmd = [sys.executable, script, "--rank", str(rank), "--world_size", str(world_size)] + seg_args
    p = subprocess.Popen(cmd, env=env)
    procs.append(p)

codes = [p.wait() for p in procs]
if any(codes):
    sys.exit(f"some ranks failed, exit codes: {codes}")
