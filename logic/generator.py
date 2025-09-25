from abc import ABC, abstractmethod
from typing import Tuple, Optional, Literal, Any, List
from pathlib import Path
import logging

from omegaconf import OmegaConf
from tqdm import tqdm

import numpy as np
import math

import torch
from torch import nn
from torch.nn import functional as F

from flow_matching.utils import ModelWrapper, categorical
from flow_matching.path import ProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.solver.solver import Solver
from flow_matching.solver.utils import get_nearest_times

from logic.flow import SourceDistribution, MaskedSourceDistribution, UniformSourceDistribution
from model.DiT import ConditionalDDiTlM

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.softmax(self.model(x, t, **kwargs).float(), -1)

class BaseGenerator(ABC):
    def __init__(
            self,
            model: nn.Module,
            ckpt_path: Path,
            batch_size: int,
            vocab_size: int,
            max_len: int,
            # logging 参数
            logging_level: Optional[Literal["DEBUG", "INFO"]] = "DEBUG",

            # Flow matching 参数
            scheduler_type: str = "polynomial",
            scheduler_exponent: Optional[float] = 2.0,
            source_distribution: str = "uniform",
            sampling_steps: int = 1024,
            time_epsilon: float = 0.0,
            diffusion_on_pad: bool = False,

    ):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.logging_level = logging_level
        self._setup_logging()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.unwrapped_model = model.to(self.device)
        self.load_checkpoint(ckpt_path)
        self.model = WrappedModel(model=self.unwrapped_model)

        self.path = self._setup_path(scheduler_type, scheduler_exponent)
        self.source_dist = self._setup_source_distribution(source_distribution)
        self.diffusion_on_pad = diffusion_on_pad
        self.add_token = 1 if source_distribution == 'mask' else 0
        self.solver = self._setup_solver(vocab_size + self.add_token)
        self.sampling_steps = sampling_steps
        self.time_epsilon = time_epsilon

    def wrap_batch(self, seqs: List[str]=None, labels: List[str]=None, seq_len: List[int]|int=None):
        assert seqs is not None or seq_len is not None

        labels = torch.tensor(np.array([label.split('.') for label in labels], dtype=int))
        if seqs is None:
            if isinstance(seq_len, int):
                seq_len = torch.ones(self.batch_size) * seq_len
            else: 
                seq_len = torch.tensor(seq_len)
            attention_mask = torch.zeros((self.batch_size, self.max_len), dtype=int)
            for b in range(self.batch_size):
                attention_mask[b, :seq_len[b]] = 1

            seqs = ['None' for _ in range(self.batch_size)]

        else:
            attention_mask = torch.zeros((self.batch_size, self.max_len), dtype=int)
            for b in range(self.batch_size):
                attention_mask[b, :len(seqs[b])] = 1
            
        return {
            'sequence': seqs, 
            'label': labels,
            'attention_mask': attention_mask
        }



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

    def _setup_solver(self, vocab_size) -> MixtureDiscreteEulerSolver:
        return MixtureDiscreteEulerSolver(self.model, self.path, vocabulary_size=vocab_size)
    
    def _setup_logging(self,):
        logging.basicConfig(
            level=self.logging_level,
            format='%(asctime)s/ %(name)s/ %(levelname)s/ %(message)s',
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_checkpoint(self, ckpt_path: Path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.logger.info(f"Loaded checkpoint at Epoch {ckpt['epoch']}, Step {ckpt['global_step']}")
        self.unwrapped_model.load_state_dict(ckpt['model_state_dict'])

    def _sample_source(self, mask: torch.Tensor) -> torch.Tensor:
        return self.source_dist.sample(mask=mask, device=self.device)
    
    def _sample(self, x_0: torch.Tensor, c: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        # print(f'x_0 dtype', x_0.dtype, 'c dtype', c.dtype, 'mask dtype', mask.dtype)
        sample = self.solver.sample(
            x_init=x_0,
            step_size=1/self.sampling_steps,
            verbose=False,
            dtype_categorical=torch.float64,
            time_grid=torch.tensor([0.0, 1.0 - self.time_epsilon]),
            return_itermediates=False,
            c=c,
            input_mask=mask
        )

        # print('sample dtype', sample.dtype)
        if mask is None:
            return sample
        else:
            return sample * mask
        
    @torch.no_grad()   
    def _self_sample(self, x_0: torch.Tensor, **model_extras) -> Tuple[torch.Tensor, torch.Tensor]:
        time_grid = torch.tensor([0.0, 1.0 - self.time_epsilon], device=self.device)

        t_init = time_grid[0].item()
        t_final = time_grid[-1].item()

        n_steps = math.ceil((t_final - t_init) * self.sampling_steps)

        t_discretization = torch.tensor(
            [t_init + 1 / self.sampling_steps * i for i in range(n_steps)] + [t_final],
                device=self.device,
        )

        x_t = x_0.clone()

        steps_counter = 0

        ctx = tqdm(total=t_final, desc=f'NFE: {steps_counter}')
        with ctx:
            for i in range(n_steps):
                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]

                p_1t = self.solver.model(x=x_t, t=t.repeat(x_t.shape[0]), **model_extras)
                x_1 = categorical(p_1t.to(dtype=torch.float64))

                if i == n_steps - 1:
                    x_t = x_1
                else:
                    scheduler_output = self.solver.path.scheduler(t=t)
                    k_t = scheduler_output.alpha_t
                    d_k_t = scheduler_output.d_alpha_t

                    delta_1 = F.one_hot(x_1, num_classes=self.vocab_size + self.add_token).to(k_t.dtype)

                    u = d_k_t / (1 - k_t) * delta_1

                    delta_t = F.one_hot(x_t, num_classes=self.vocab_size + self.add_token)
                    u = torch.where(
                        delta_t.to(dtype=torch.bool), torch.zeros_like(u), u
                    )

                    intensity = u.sum(dim=-1)
                    mask_jump = torch.rand(
                        size=x_t.shape, device=self.device
                    ) < 1 - torch.exp(-h * intensity)

                    if mask_jump.sum() > 0:
                        x_t[mask_jump] = categorical(
                            u[mask_jump].to(dtype=torch.float64)
                        )

                steps_counter += 1
                t = t + h

                ctx.n = t.item()
                ctx.refresh()
                ctx.set_description(f"NFE: {steps_counter}")

        return x_t, p_1t
        
        
    def _decode_batch(self, x_t: torch.Tensor):
        x_t_indices = x_t.tolist()
        decode_strs = []
    
        for x_t_ids in x_t_indices:
            tokens = [list('*ACDEFGHIKLMNPQRSTVWY?')[id] for id in x_t_ids]
            decode_strs.append(''.join([tok for tok in tokens if tok != '*']))

        return decode_strs
    
    @abstractmethod
    def _get_batch_data(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass
    
    def _generate(self, batch):
        raw_seqs, mask, labels = self._get_batch_data(batch)
        
        x_0 = self._sample_source(mask)

        if self.diffusion_on_pad:
            _mask = torch.ones_like(mask, device=mask.device)
        else:
            _mask = mask
        # print('before sample', _mask.dtype)
        
        # sample = self._sample(x_0, labels, _mask)
        sample, logits = self._self_sample(x_0, c=labels, input_mask=_mask)
        sample = sample * _mask

        gen_seqs = self._decode_batch(sample)

        # with open('generation_result.txt', 'w') as f:
        #     for i, (g_seq, r_seq) in enumerate(zip(gen_seqs, raw_seqs)):
        #         f.write(f'label:  {labels[i].tolist()} \n')
        #         f.write(f'raw: {r_seq[:self.max_len]} \n')
        #         f.write(f'gen: {g_seq} \n')
        # f.close()
        return sample, logits, labels, gen_seqs, raw_seqs
    
    def compute_perplexity(self, sample: torch.Tensor, logits: torch.Tensor):
        assert sample.shape == logits.shape[:2]
        cross_entropy = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), sample.reshape(-1), reduction='none').reshape(sample.shape)

        mask = (sample != 0).float()
        valid_tokens = mask.sum(-1)

        averge_loss = (cross_entropy * mask).sum(-1) / valid_tokens

        perplexity = torch.exp(averge_loss)
        return perplexity


    def generate(self, batch, num_rounds: int = 10, result_file: str = 'generation_result.txt'):
        with open(result_file, 'a') as f:
            for _ in range(num_rounds):
                sample, logits, labels, gen_seqs, raw_seqs = self._generate(batch)
                perplexity = self.compute_perplexity(sample, logits)
                for i, (g_seq, r_seq) in enumerate(zip(gen_seqs, raw_seqs)):
                    f.write(f'label:  {labels[i].tolist()} \n')
                    f.write(f'raw: {r_seq[:self.max_len]} \n')
                    f.write(f'gen: {g_seq} \n')
                    f.write(f'gen ppl: {perplexity[i].item()} \n')
        f.close()
            



        


    


class FlowMatchingGenerator(BaseGenerator):
    def __init__(self, cfg: OmegaConf):
        add_token = 1 if cfg.flow.source_distribution == "mask" else 0

        if cfg.model.type == 'DDiT':
            model = ConditionalDDiTlM(
                n_tokens=cfg.data.vocab_size + add_token,
                num_classes=cfg.data.num_classes,
                model_config=cfg.model,
                pad_token_id=cfg.data.pad_token_id,
            )
        super().__init__(
            model=model,
            ckpt_path=cfg.valid.ckpt_path,
            batch_size=cfg.valid.batch_size,
            vocab_size=cfg.data.vocab_size,
            max_len=cfg.data.max_len,
            logging_level=cfg.log.level,
            scheduler_type=cfg.flow.scheduler_type,
            scheduler_exponent=cfg.flow.scheduler_exponent,
            source_distribution=cfg.flow.source_distribution,
            time_epsilon=cfg.flow.time_epsilon,
            diffusion_on_pad=cfg.flow.diffusion_on_pad,
        )
    
    def _get_batch_data(self, batch):
        mask, sequences, label = batch['attention_mask'].to(self.device), batch['sequence'], batch['label'].to(self.device)

        return sequences, mask, label
    


        
