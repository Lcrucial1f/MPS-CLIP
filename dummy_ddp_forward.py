import os, argparse, torch, torch.distributed as dist
from ptflops import get_model_complexity_info
from ruamel.yaml import YAML
from models.model_retrieval import HarMA   # ← 换成你的路径

# ------------- 超参数 -------------
BATCH        = 64         # 真实推理/训练时的 batch
BATCH_FLOPS  = 64         # 计算 FLOPs 时的 batch，显存紧张可改 1
IMG_SHAPE    = (3, 224, 224)
TXT_LEN      = 77


# ---------- 分布式初始化 ----------
def init_dist(args):
    # 单卡
    if "RANK" not in os.environ:
        args.rank = args.gpu = 0
        args.world_size = 1
        args.distributed = False
        return

    args.rank        = int(os.environ["RANK"])
    args.world_size  = int(os.environ["WORLD_SIZE"])
    args.gpu         = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.gpu)

    dist.init_process_group("nccl")
    args.distributed = True
    dist.barrier()


# ------------- ptflops ----------
def flops_and_params(cfg, device):
    def input_constructor(_):
        img = torch.randn(BATCH_FLOPS, *IMG_SHAPE, device=device)
        txt = torch.randint(0, cfg.get("vocab_size", 49408),
                            (BATCH_FLOPS, TXT_LEN), device=device)
        return dict(image=img, text_ids=txt)

    model_cuda = HarMA(cfg).to(device).eval()

    macs, params = get_model_complexity_info(
        model_cuda, IMG_SHAPE,
        as_strings=True,
        print_per_layer_stat=False,
        input_constructor=input_constructor,
        verbose=False            # 不逐层打印
    )
    return macs, params


# ------------------- main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--calc_flops", action="store_true")
    args = ap.parse_args()

    init_dist(args)

    # ---------------- CUDA / CPU ----------------
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，无法按要求在 GPU 上运行！")
    device = torch.device(f"cuda:{args.gpu}")

    # ---------------- 读取 yaml -----------------
    with open(args.config) as f:
        cfg = YAML(typ="safe").load(f)

    # ---------------- 创建模型 ------------------
    model = HarMA(cfg).to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])

    # ---------- rank-0 统计 FLOPs --------------
    if args.calc_flops and args.rank == 0:
        macs, params = flops_and_params(cfg, device)
        # 若 BATCH_FLOPS ≠ 实际 batch，可按需提示
        print(f"[ptflops] Params: {params}  "
              f"FLOPs (batch={BATCH_FLOPS}): {macs}")

    # ------------- 构造输入并前向 ---------------
    img = torch.randn(BATCH, *IMG_SHAPE, device=device)
    txt = torch.randint(0, cfg.get("vocab_size", 49408),
                        (BATCH, TXT_LEN), device=device)
    with torch.no_grad():
        out = model(img, txt)

    if args.rank == 0:
        shape = ([x.shape for x in out] if isinstance(out, (list, tuple))
                 else out.shape)
        print("[forward] output shape(s):", shape)

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
