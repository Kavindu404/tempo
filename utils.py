import os
import random
import numpy as np
import torch
import torch.distributed as dist

def mkdir_p(path: str):
    os.makedirs(path, exist_ok=True)

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process():
    return get_rank() == 0

def init_distributed_mode(backend="nccl"):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # single-process fallback
        rank = 0
        world_size = 1
        gpu = 0
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend=backend, init_method="env://")
    dist.barrier()
    return rank, world_size, gpu

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def reduce_dict(input_dict, average=True):
    if not is_dist_avail_and_initialized():
        return input_dict
    with torch.no_grad():
        names, values = [], []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= dist.get_world_size()
        reduced = {k: v for k, v in zip(names, values)}
    return reduced

def denorm_img(img_t, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    # img_t: (3,H,W), 0-1 normalized with mean/std
    img = img_t.detach().cpu().numpy()
    img = (img.transpose(1,2,0) * std) + mean
    img = np.clip(img*255.0, 0, 255).astype(np.uint8)
    return img
