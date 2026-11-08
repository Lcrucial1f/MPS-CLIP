import subprocess
import os

world_size = 7
script = "segmentation.py"

procs = []
for rank in range(world_size):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(rank)
    cmd = ["python", script, "--rank", str(rank), "--world_size", str(world_size)]
    p = subprocess.Popen(cmd, env=env)
    procs.append(p)

for p in procs:
    p.wait()
