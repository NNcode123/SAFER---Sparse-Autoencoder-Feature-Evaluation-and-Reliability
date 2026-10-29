from dataclasses import dataclass
from typing import Callable

from torch import Tensor


PenaltyFunction = Callable[[Tensor], Tensor]


def l1_penalty(activations: Tensor) -> Tensor:
    return activations.abs().mean()


def l2_penalty(activations: Tensor) -> Tensor:
    return activations.square().mean()


PENALTY_FUNCTIONS: dict[str, PenaltyFunction] = {
    "l1": l1_penalty,
    "l2": l2_penalty,
}


@dataclass
class TrainingConfig:
    input_output_features: int
    intermediate_features: int
    regularization_penalty: str
    intermediate_activation: str = "ReLU"
    version: str = "1.0"
