
from abc import ABC
from typing import Tuple, Optional

import torch
from torch.nn.modules.loss import _Loss

from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import MixtureDiscreteProbPath, ProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler


class SourceDistribution(ABC):
    def __init__(
            self,
    ) -> None:
        pass

    def sample(self, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        pass
    # def sample_like(self, : torch.Tensor) -> torch.Tensor:
    #     pass


class MaskedSourceDistribution(SourceDistribution):
    def __init__(self, mask_token: int) -> None:
        self.mask_token = mask_token

    @property
    def masked(self) -> bool:
        return True
    
    def sample(self, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        return torch.zeros_like(mask).fill_(self.mask_token).long()

class UniformSourceDistribution(SourceDistribution):
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    @property
    def masked(self) -> bool:
        return False
    
    def sample(self, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        return torch.randint_like(mask, high=self.vocab_size)