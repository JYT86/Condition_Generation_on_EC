from abc import ABC, abstractmethod
from typing import Optional, Literal, Any, Tuple
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm
import logging

import math

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from flow_matching.path import ProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.loss import MixturePathGeneralizedKL

from data.data import build_dataloaders
from logic.flow import SourceDistribution, MaskedSourceDistribution, UniformSourceDistribution
from model.DiT import ConditionalDDiTlM
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import StateDictType, FullStateDictConfig



class BaseTrainer(ABC):
    def __init__(
        self,
        rank: int, 
        world_size: int,

        vocab_size: int,
        model: nn.Module,
        optimizer: Optimizer,
        optimizer_kwargs: dict,
        data_name: str,
        file_path: str, 
        max_len: int,
        batch_size: int,

        # optimization 参数
        lr: float = 1e-3,
        warmup: int = 100,
        n_iters: int = 1000,
        eta_min_ratio: float = 0.01,
        grad_clip: float = 0.0,
        accumulation_step: int = 4,

        # ckpt 参数
        save_ckpt_step: int = 500,
        ckpt_dir: str = './checkpoints',

        # logging 参数
        logging_level: Optional[Literal["DEBUG", "INFO"]] = "DEBUG",
        logging_freq_step: int = 100,

        # flow-matching 参数
        scheduler_type: str = "polynomial",
        scheduler_exponent: Optional[float] = 2.0,
        source_distribution: str = "uniform",
        loss_func: str = "cross_entropy",
        time_epsilon: float = 0.0,
        diffusion_on_pad: bool = True,
            
    ):
        self.rank = rank
        self.world_size = world_size
        
        self.device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')
        self.is_main_process = self.rank == 0

        self.vocab_size = vocab_size

        # 分布式
        self.model = model.to(self.device)
        if self.world_size > 1:
            #auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=1e6)
            self.model = FSDP(
                self.model,
                device_id=torch.device(f"cuda:{self.rank}"),
            )
        if isinstance(optimizer, type):
            kwargs = optimizer_kwargs or {}
            self.optimizer = optimizer(self.model.parameters(), **kwargs)
        else:
            # if user passed an optimizer instance, re-create it bound to FSDP params
            # (try to preserve lr if possible)
            try:
                # try to extract lr from existing optimizer
                lr = optimizer.param_groups[0]['lr']
            except Exception:
                lr = 1e-3
            self.optimizer = type(optimizer)(self.model.parameters(), lr=lr)
        self.loaders = build_dataloaders(
            rank, world_size, data_name, file_path, max_len, world_size > 1, batch_size, batch_size
        )

        # 初始化 optimization 参数
        self.lr = lr
        self.warmup = warmup
        self.n_iters = n_iters
        self.eta_min_ratio = eta_min_ratio
        self.grad_clip = grad_clip
        self.accumulation_step = accumulation_step

        self.save_ckpt_step = save_ckpt_step
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(exist_ok=True)

        # 初始化 训练状态
        self.step = 0
        self.epoch = 0

        # 初始化 Flow Matching 组件
        self.diffusion_on_pad = diffusion_on_pad
        self.path = self._setup_path(scheduler_type, scheduler_exponent)
        self.source_dist = self._setup_source_distribution(source_distribution)
        self.loss_function = self._setup_loss_function(loss_func)
        self.time_epsilon = time_epsilon

        # 初始化 logger
        self.logging_level = logging_level
        self.logging_feq_step = logging_freq_step
        if self.is_main_process:
            self._setup_logging()
            self.logger.info(f"Flow Matching Trainer initialized with:")
            self.logger.info(f"- Scheduler: {scheduler_type} (exponent: {scheduler_exponent})")
            self.logger.info(f"- Source distribution: {source_distribution}")
    
    def _setup_logging(self,):
        logging.basicConfig(
            level=self.logging_level,
            format='%(asctime)s/ %(name)s/ %(levelname)s/ %(message)s',
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def _setup_path(self, scheduler_type: str, exponent: Optional[float]) -> ProbPath:
        if scheduler_type == "poliynomial":
            scheduler = PolynomialConvexScheduler(n=exponent or 1.0)
        else:
            raise ValueError(f"{scheduler_type} is not supported")
        
        return MixtureDiscreteProbPath(scheduler=scheduler)
    
    def _setup_source_distribution(self, source_distribution: str) -> SourceDistribution:
        if source_distribution == "mask":
            return MaskedSourceDistribution(mask_token=self.vocab_size)
        elif source_distribution == "uniform":
            return UniformSourceDistribution(vocab_size=self.vocab_size)
        else:
            raise ValueError(f"{source_distribution} is not supported")
    
    def _setup_loss_function(self, loss_func: str) -> nn.Module:
        """设置损失函数"""
        if loss_func == 'cross_entropy':
            return torch.nn.CrossEntropyLoss(ignore_index=-100 if self.diffusion_on_pad else 0)
        elif loss_func == 'generalized_kl':
            return MixturePathGeneralizedKL(path=self.path)
        else:
            raise ValueError(f"{loss_func} is not supported") 
        
    def _get_lr(self) -> float:
        if self.step < self.warmup:
            return self.lr * (self.step / self.warmup)
        
        else:
            total_steps = self.n_iters
            eta_min = self.eta_min_ratio * self.lr
            cosine_decay = 0.5 * (
                1 + math.cos(math.pi * (self.step - self.warmup) / (total_steps - self.warmup))
            )
            return eta_min + (self.lr - eta_min) * cosine_decay
    
    def _setup_lr(self, ) -> float:
        lr = self._get_lr()
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr
    
    @abstractmethod
    def get_batch_data(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从batch中提取token_ids, mask, labels
        """
        pass

    def sample_time(self, bsz: int) -> torch.Tensor:
        return torch.rand(bsz, device=self.device) * (1 - self.time_epsilon)
    
    def sample_source(self, mask: torch.Tensor) -> torch.Tensor:
        return self.source_dist.sample(mask=mask, device=self.device)
    
    def compute_flow_matching_loss(
            self, x_0: torch.Tensor, x_1: torch.Tensor,
            label: torch.Tensor, t: torch.Tensor, mask: torch.Tensor,
    )-> torch.Tensor:
        path_sample = self.path.sample(x_0, x_1, t)
        
        if self.diffusion_on_pad:
            _mask = torch.ones_like(mask, device=mask.device)
        else:
            _mask = mask
        # self.logger.debug(f'time tensor: {t}, {path_sample.t}')
        logits = self.model(x=path_sample.x_t, t=path_sample.t, c=label, input_mask=_mask)
        if isinstance(self.loss_function, nn.CrossEntropyLoss):
            loss = self.loss_function(logits.flatten(0,1), x_1.flatten(0, 1)).mean()
        elif isinstance(self.loss_function, MixturePathGeneralizedKL):
            loss = self.loss_function(
                logits=logits, x_1=x_1, x_t=path_sample.x_t*_mask, t=path_sample.t
            ).mean()
        else:
            raise ValueError("Invalid loss func")
        return loss
    
    def optimization_step(self, loss: torch.Tensor):
        lr = self._setup_lr()
        loss /= self.accumulation_step
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )
        if (self.step + 1) % self.accumulation_step == 0:
            
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        if self.is_main_process:
            self.logger.debug('finish optimization step')
        return {
            'lr': lr
        }

    def train_step(self, batch: Any):
        x_1, mask, label = self.get_batch_data(batch)
        assert x_1.shape == mask.shape and label.shape[0] == x_1.shape[0], f"label.shape: {label.shape}, x_1.shape {x_1.shape}, mask.shape {mask.shape}"

        bsz = x_1.shape[0]

        # 采样 时间步 和 x_0
        t = self.sample_time(bsz)
        x_0 = self.sample_source(mask)

        # 计算 loss
        loss = self.compute_flow_matching_loss(x_0, x_1, label, t, mask)
        lr = self.optimization_step(loss)['lr']

        self.step += 1
        return {
            'loss': loss.item(),
            'lr': lr
        }
    
    def train_epoch(self):
        self.model.train()

        train_sampler = self.loaders.get('train_sampler', None)
        if train_sampler:
            train_sampler.set_epoch(self.epoch)
        train_loader = self.loaders['train_loader']

        if self.is_main_process:
            bar = tqdm(total=len(train_loader))

        total_loss = 0.0
        total_count = 0

        for batch_idx, batch in enumerate(train_loader):
            train_info = self.train_step(batch)
            total_loss += train_info['loss']
            total_count += 1

            if self.is_main_process: 
                bar.update(1)
                bar.set_postfix(loss=train_info['loss'], lr=train_info['lr'])
            
            if self.step % self.save_ckpt_step == 0:
                self.save_ckpt()

        self.epoch += 1

            
        if self.is_main_process:
            bar.close()
        
        if self.world_size > 1:
            total_loss_tensor = torch.tensor([total_loss]).to(self.device)
            count_tensor = torch.tensor([total_count]).to(self.device)

            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

            mean_loss = total_loss_tensor / count_tensor
        else:
            mean_loss = total_loss / total_count

        if self.is_main_process:
            self.logger.info(f'Epoch {self.epoch}, mean loss: {mean_loss.item():.3f}')

    def save_ckpt(self):
        """保存ckpt"""
        if self.is_main_process:
            if isinstance(self.model, FSDP):
                with FSDP.state_dict_type(
                    self.model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                ):
                    model_state = self.model.state_dict()
            else:
                model_state = self.model.state_dict()
        
            ckpt = {
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'global_step': self.step,
                'epoch': self.epoch,
            }

            ckpt_path = self.ckpt_dir / f'ckpt_e_{self.epoch}_s_{self.step}.pt'
            torch.save(ckpt, ckpt_path)

            self.logger.info(f'Checkpoint saved at epoch {self.epoch}, step {self.step}')

    def load_ckpt(self, ckpt_path: Path):
        """加载ckpt"""
        if not ckpt_path.exists():
            if self.is_main_process:
                self.logger.info("No checkpoint found, starting from scratch")
            return
        
        ckpt = torch.load(ckpt_path, map_location=self.device)
        # 加载模型状态
        if isinstance(self.model, FSDP):
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
            ):
                self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt['model_state_dict'])
        # 加载优化器状态
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        self.epoch = ckpt['epoch']
        self.step = ckpt['global_step']
        
        if self.is_main_process:
            self.logger(f"Loaded checkpoint: epoch {self.epoch}, step {self.step}")

    def train(self, epochs):
        for epoch in range(epochs):
            self.train_epoch()


class FlowMatchingTrainer(BaseTrainer):
    def __init__(self, rank: int, world_size: int, cfg: OmegaConf):
        add_token = 1 if cfg.flow.source_distribution == 'mask' else 0

        if cfg.model.type == 'DDiT':
            model = ConditionalDDiTlM(
                n_tokens=cfg.data.vocab_size + add_token,
                num_classes=cfg.data.num_classes,
                model_config=cfg.model,
                pad_token_id=cfg.data.pad_token_id,
            )
        optimizer_cls = torch.optim.Adam
        optimizer_kwargs = {"lr": float(cfg.optim.lr)}
        super().__init__(
            rank=rank,
            world_size=world_size,
            vocab_size=cfg.data.vocab_size,
            model=model,
            optimizer=optimizer_cls, 
            optimizer_kwargs=optimizer_kwargs,
            data_name=cfg.data.name,
            file_path=cfg.data.file_path,
            max_len=cfg.data.max_len,
            batch_size=cfg.train.batch_size,
            lr=cfg.optim.lr,
            warmup=cfg.optim.warmup,
            n_iters=cfg.optim.n_iters,
            eta_min_ratio=cfg.optim.eta_min_ratio,
            grad_clip=cfg.optim.grad_clip,
            accumulation_step=cfg.optim.accumulation_step,
            save_ckpt_step=cfg.train.save_ckpt_step,
            ckpt_dir=cfg.train.ckpt_dir,
            logging_level=cfg.log.level,
            logging_freq_step=cfg.log.freq_step,
            scheduler_type=cfg.flow.scheduler_type,
            scheduler_exponent=cfg.flow.scheduler_exponent,
            source_distribution=cfg.flow.source_distribution,
            loss_func=cfg.flow.loss_func,
            time_epsilon=cfg.flow.time_epsilon,
            diffusion_on_pad=cfg.flow.diffusion_on_pad,
        )
    
    def get_batch_data(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        label = batch['label'].to(self.device)

        return input_ids, attention_mask, label
        