import torch 
import torch.nn as nn
from typing import Tuple, List 
import numpy as np 

def quantized_weights(weights: torch.Tensor) -> Tuple[torch.Tensor, float]:
    '''
    Quantize the weights so that all values are integers between -128 and 127.
    You may want to use the total range, 3-sigma range, or some other range when
    deciding just what factors to scale the float32 values by.

    Parameters:
    weights (Tensor): The unquantized weights

    Returns:
    (Tensor, float): A tuple with the following elements:
                        * The weights in quantized form, where every value is an integer between -128 and 127.
                          The "dtype" will still be "float", but the values themselves should all be integers.
                        * The scaling factor that your weights were multiplied by.
                          This value does not need to be an 8-bit integer.
    '''
    inv_scale = (1+1) / (torch.max(weights)-torch.min(weights))
    zero_point = 0

    result = (weights * inv_scale).round() + zero_point
    return torch.clamp(result, min=-1, max=1), inv_scale

#recursively searches for linear and conv layers
def quantize_layer_weights(model: nn.Module, config):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            q_layer_data, scale = quantized_weights(module.weight.data)
            q_layer_data = q_layer_data.to(config.device)
            module.weight.data = q_layer_data
            module.weight.scale = scale

            if (q_layer_data < -128).any() or (q_layer_data > 127).any():
                raise Exception("Quantized weights of {} layer include values out of bounds for an 8-bit signed integer".format(name))
            if (q_layer_data != q_layer_data.round()).any():
                raise Exception("Quantized weights of {} layer include non-integer values".format(name))
        else:
            quantize_layer_weights(module)

#generalized hooking

def register_activation_profiling_hooks(model: nn.Module):
    model.input_activations = np.empty(0)
    model.layer_activations = {}

    model.profile_activations = True

    def activations_hook(name):
        def hook(layer, x, y):
            if model.profile_activations:
                model.layer_activations[name] = np.append(model.layer_activations.get(name, np.empty(0)), x[0].cpu().view(-1))
        return hook

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer.register_forward_hook(activations_hook(name))


class NetQuantized(nn.Module):
    def __init__(self, net: nn.Module):
        super(NetQuantized, self).__init__()
        self.net = net
        self.quantized_layers = []

        # Identify and process each quantizable layer
        for name, module in self.net.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.quantized_layers.append(module)
                self.register_pre_hooks(module)

        self.input_activations = net.input_activations
        self.input_scale = self.quantize_initial_input(self.input_activations)
        self.setup_output_scales()

    def register_pre_hooks(self, module):
        # Define and register a forward pre-hook
        def pre_hook(layer, x):
            x = x[0]
            if (x < -128).any() or (x > 127).any():
                raise Exception("Input to {} layer is out of bounds for an 8-bit signed integer".format(layer.__class__.__name__))
            if (x != x.round()).any():
                raise Exception("Input to {} layer has non-integer values".format(layer.__class__.__name__))
        module.register_forward_pre_hook(pre_hook)

    def setup_output_scales(self):
        # Calculate and set the output scales and quantize biases for quantizable layers
        preceding_layer_scales = []
        for layer in self.quantized_layers:
            layer.output_scale = self.quantize_activations(layer.activations, layer.weight.scale, self.input_scale, preceding_layer_scales)
            preceding_layer_scales.append((layer.weight.scale, layer.output_scale))

            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data = self.quantized_bias(
                    layer.bias.data,
                    layer.weight.scale,
                    self.input_scale,
                    preceding_layer_scales
                )
                if (layer.bias.data < -2147483648).any() or (layer.bias.data > 2147483647).any():
                    raise Exception("Bias for layer {} has values which are out of bounds for a 32-bit signed integer".format(layer.__class__.__name__))
                if (layer.bias.data != layer.bias.data.round()).any():
                    raise Exception("Bias for layer {} has non-integer values".format(layer.__class__.__name__))

    @staticmethod
    def quantize_initial_input(pixels: np.ndarray) -> float:
        return (127 + 128) / (np.max(pixels) - np.min(pixels))

    @staticmethod
    def quantize_activations(activations: np.ndarray, n_w: float, n_initial_input: float, ns: List[Tuple[float, float]]) -> float:
        cumulative_scale = n_initial_input * n_w
        for weight_scale, output_scale in ns:
            cumulative_scale *= output_scale * weight_scale
        return (127 + 128) / (np.max(activations) - np.min(activations)) / cumulative_scale

    @staticmethod
    def quantized_bias(bias: torch.Tensor, n_w: float, n_initial_input: float, ns: List[Tuple[float, float]]) -> torch.Tensor:
        cumulative_scale = n_initial_input * n_w
        for weight_scale, output_scale in ns:
            cumulative_scale *= output_scale * weight_scale
        return torch.clamp((bias * cumulative_scale).round(), min=-2147483648, max=2147483647)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply initial scaling
        x = torch.clamp(torch.round(x * self.input_scale), min=-128, max=127)

        # Process each module respecting the original network structure
        modules = list(self.net.named_children())
        for name, module in modules:
            x = module(x)
            if module in self.quantized_layers and hasattr(module, 'output_scale'):
                x = torch.clamp(torch.round(x * module.output_scale), min=-128, max=127)

        return x
