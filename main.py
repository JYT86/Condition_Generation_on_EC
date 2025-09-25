import os
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from logic.trainer import FlowMatchingTrainer
from data.data import build_dataloaders

def setup():
    """初始化分布式训练环境"""
    # 从环境变量获取rank信息
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # 设置当前GPU设备
    torch.cuda.set_device(local_rank)
    
    # 初始化进程组
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    
    print(f"Process {rank}/{world_size-1} initialized on GPU {local_rank}")
    
    return rank, world_size, local_rank

def cleanup():
    dist.destroy_process_group()

def main():
    # 初始化分布式环境并获取rank信息
    rank, world_size, local_rank = setup()
    
    # 加载配置
    cfg = OmegaConf.load("config/config.yaml")
    
    # 创建训练器，传入rank和world_size参数
    fm_trainer = FlowMatchingTrainer(
        rank=rank,
        world_size=world_size,
        cfg=cfg
    )
    
    # 开始训练
    fm_trainer.train(10)
    
    cleanup()

if __name__ == "__main__":
    main()