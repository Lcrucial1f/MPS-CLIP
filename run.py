import argparse
import os
import subprocess
import sys


TASK_CONFIGS = {
    'itr_rsicd_vit': 'configs/Retrieval_rsicd_vit.yaml',
    'itr_rsitmd_vit': 'configs/Retrieval_rsitmd_vit.yaml',
    'itr_rsicd_geo': 'configs/Retrieval_rsicd_geo.yaml',
    'itr_rsitmd_geo': 'configs/Retrieval_rsitmd_geo.yaml',
}


def build_distributed_command(args):
    env = os.environ.copy()
    if args.dist.startswith('gpu'):
        gpu_index = int(args.dist[3:])
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
        world_size = 1
    elif args.dist.startswith('f') and args.dist[1:].isdigit():
        world_size = int(args.dist[1:])
        if world_size < 1:
            raise ValueError('The process count in --dist must be positive')
    else:
        raise ValueError("--dist must be 'fN' (N processes) or 'gpuN' (one selected GPU)")

    command = [
        sys.executable,
        '-W',
        'ignore',
        '-m',
        'torch.distributed.run',
        '--nnodes=1',
        f'--nproc_per_node={world_size}',
        f'--master_port={args.master_port}',
        'Retrieval.py',
        '--config',
        args.config,
        '--output_dir',
        args.output_dir,
        '--bs',
        str(args.bs),
        '--checkpoint',
        args.checkpoint,
    ]
    if args.evaluate:
        command.append('--evaluate')
    return command, env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=sorted(TASK_CONFIGS), default='itr_rsitmd_vit')
    parser.add_argument('--dist', default='f2', help="'fN' uses N visible GPUs; 'gpuN' selects GPU N")
    parser.add_argument('--config', help='override the config selected by --task')
    parser.add_argument('--bs', default=-1, type=int, help='global training batch size override')
    parser.add_argument('--checkpoint', default='-1', help='checkpoint used for resume or evaluation')
    parser.add_argument('--output_dir', default='./outputs/test')
    parser.add_argument('--master_port', default=29500, type=int)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()

    if args.config is None:
        args.config = TASK_CONFIGS[args.task]

    os.makedirs(args.output_dir, exist_ok=True)
    command, env = build_distributed_command(args)
    subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    main()
