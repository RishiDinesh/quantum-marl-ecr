from collections import OrderedDict
from typing import Any, List

import torch
import torch.nn as nn

import pennylane as qml

from maro.rl.model.fc_block import FullyConnected


def encode(n_qubits, inputs):
    for wire in range(n_qubits):
        qml.RX(inputs[wire], wires=wire)


def layer(n_qubits, y_weight, z_weight):
    for wire, y_weight in enumerate(y_weight):
        qml.RY(y_weight, wires=wire)
    for wire, z_weight in enumerate(z_weight):
        qml.RZ(z_weight, wires=wire)
    for wire in range(n_qubits):
        qml.CZ(wires=[wire, (wire + 1) % n_qubits])


def measure(n_qubits):
    return [
        qml.expval(qml.PauliZ(qubit)) for qubit in range(n_qubits)
    ]


def get_quantum_net(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)
    shapes = {
        "y_weights": (n_layers, n_qubits),
        "z_weights": (n_layers, n_qubits)
    }

    @qml.qnode(dev, interface='torch')
    def circuit(inputs, y_weights, z_weights):
        for layer_idx in range(n_layers):
            if layer_idx == 0:
                encode(n_qubits, inputs)
            layer(n_qubits, y_weights[layer_idx], z_weights[layer_idx])
        return measure(n_qubits)

    model = qml.qnn.TorchLayer(circuit, shapes)

    return model


class ATan(nn.Module):
    '''
    Applies the Arc-Tangent (ATan) function element-wise:
        ATan(x) = arctan(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = ATan()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''

    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return torch.arctan(input)


class QuantumFullyConnected(nn.Module):
    """Quantum Fully connected network with optional batch normalization, activation and dropout components.
    Args:
        name (str): Network name.
        input_dim (int): Network input dimension.
        output_dim (int): Network output dimension.
        hidden_dims (List[int]): Dimensions of hidden layers. Its length is the number of hidden layers.
        n_qubits (int): Number of qubits.
        n_layers (int): VQC network layers.
        activation: A ``torch.nn`` activation type. If None, there will be no activation. Defaults to LeakyReLU.
        head (bool): If true, this block will be the top block of the full model and the top layer of this block
            will be the final output layer. Defaults to False.
        softmax (bool): If true, the output of the net will be a softmax transformation of the top layer's
            output. Defaults to False.
        batch_norm (bool): If true, batch normalization will be performed at each layer.
        skip_connection (bool): If true, a skip connection will be built between the bottom (input) layer and
            top (output) layer. Defaults to False.
        dropout_p (float): Dropout probability. Defaults to None, in which case there is no drop-out.
        gradient_threshold (float): Gradient clipping threshold. Defaults to None, in which case not gradient clipping
            is performed.
    """

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        n_qubits: int = 4,
        n_layers: int = 6,
        activation=nn.LeakyReLU(),
        head: bool = False,
        softmax: bool = False,
        batch_norm: bool = False,
        skip_connection: bool = False,
        dropout_p: float = None,
        gradient_threshold: float = None,
        name: str = None
    ):
        super().__init__()

        self._name = name
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims if hidden_dims is not None else []
        self._output_dim = output_dim
        self._softmax = nn.Softmax(dim=1) if softmax else None

        self._n_qubits = n_qubits
        self._n_layers = n_layers

        layers = []
        layers.append(FullyConnected(input_dim,
                                     hidden_dims[-1],
                                     hidden_dims[:-1],
                                     activation,
                                     head,
                                     False,
                                     batch_norm,
                                     skip_connection,
                                     dropout_p,
                                     gradient_threshold))
        layers.append(self._build_quantum_net())

        self._net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._net(x)
        return self._softmax(out) if self._softmax else out

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _build_quantum_net(self) -> nn.Module:
        """Build a quantum circuit layer.

        Linear -> Activation -> QVC -> Linear
        """

        components = []
        components.append(("shaper", nn.Linear(
            self._hidden_dims[-1], self._n_qubits)))
        components.append(("activation", ATan()))
        components.append(
            ("qvc", get_quantum_net(self._n_qubits, self._n_layers)))
        components.append(("linear", nn.Linear(
            self._n_qubits, self._output_dim)))

        return nn.Sequential(OrderedDict(components))
