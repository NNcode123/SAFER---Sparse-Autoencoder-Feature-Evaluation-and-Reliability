from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class SAEConfig:
    input_output_features: int
    intermediate_features: int
    intermediate_activation: str = "ReLU"
    version: str = "1.0"


class SAE(nn.Module):
    def __init__(self,config: SAEConfig) -> None:

        super().__init__()

        self.encode_layer = nn.Linear(config.input_output_features,config.intermediate_features)

        self.activation_func = nn.ReLU()

        self.decode_layer = nn.Linear(config.intermediate_features, config.input_output_features)

    def encode(self, inputs: Tensor) -> Tensor:
        
        return self.activation_func(self.encode_layer(inputs))

    def decode(self, features: Tensor) -> Tensor: 

        return self.decode_layer(features)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:

        return self.decode_layer(self.activation_func(self.encode_layer(inputs)))
