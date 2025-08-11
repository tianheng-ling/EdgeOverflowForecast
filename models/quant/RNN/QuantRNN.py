from pathlib import Path

import torch
import torch.nn as nn

from elasticai.creator.nn.integer.stackedrnn import StackedRNN
from elasticai.creator.nn.integer.linear import Linear
from elasticai.creator.nn.integer.sequential import Sequential
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class QuantRNN(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        num_layers = kwargs.get("num_rnn_layers")
        num_in_features = kwargs.get("num_in_features")
        hidden_size = kwargs.get("hidden_size")
        window_size = kwargs.get("window_size")
        batch_size = kwargs.get("batch_size")
        num_out_features = kwargs.get("num_out_features")

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.do_int_forward = kwargs.get("do_int_forward")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")

        self.layers = nn.ModuleList()

        self.layers.append(
            StackedRNN(
                name="stackedrnn_0",
                cell_type="lstm",
                inputs_size=num_in_features,
                window_size=window_size,
                hidden_size=hidden_size,
                batch_size=batch_size,
                num_layers=num_layers,
                quant_bits=self.quant_bits,
                quant_data_dir=self.quant_data_dir,
                device=device,
            )
        )

        # support one-step ahead forecasting
        self.layers.append(
            Linear(
                name="linear_0",
                in_features=hidden_size,
                out_features=num_out_features,
                bias=True,
                quant_bits=self.quant_bits,
                quant_data_dir=self.quant_data_dir,
                device=device,
            )
        )
        self.sequential = Sequential(
            *self.layers,
            name=self.name,
            quant_data_dir=self.quant_data_dir,
        )

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:

        if self.do_int_forward:

            self.sequential.precompute()
            # quantize inputs
            inputs = inputs.to("cpu")
            q_inputs = self.sequential.quantize_inputs(inputs)
            save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

            q_outputs = self.sequential.int_forward(q_inputs)
            # save quantized outputs
            save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")

            # dequantize outputs
            dq_outputs = self.sequential.dequantize_outputs(q_outputs)
            return dq_outputs.unsqueeze(2)

        else:
            outputs = self.sequential.forward(inputs)
            return outputs.unsqueeze(2)
