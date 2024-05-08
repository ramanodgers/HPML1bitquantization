import torch 
import torch.nn as nn
from typing import Tuple, List 
import numpy as np 
import matplotlib.pyplot as plt

def quantized_weights(max, min, weights: torch.Tensor) -> Tuple[torch.Tensor, float]:
    inv_scale = (max-min) / (torch.max(weights)-torch.min(weights))
    zero_point = (min-torch.min(weights)*inv_scale).round()
    # zero_point = 0

    result = (weights * inv_scale).round() + zero_point
    return torch.clamp(result, min=min, max=max), inv_scale

def dequant_out(max,min,model: nn.Module):
    def output_hook(module, input, output):
        output = output/ module.weight.scale
        return output
    model.classifier.register_forward_hook(output_hook)

def quantize_layer_weights(max, min, model: nn.Module, device):
    for name, module in model.named_modules():
        if hasattr(module, 'weight')  :
            q_layer_data, scale = quantized_weights(max, min, module.weight.data)
            q_layer_data = q_layer_data.to(device)
            module.weight.data = q_layer_data
            module.weight.scale = scale
            if (q_layer_data < min).any() or (q_layer_data > max).any():
                raise Exception("Quantized weights of {} layer include values out of bounds for an 8-bit signed integer".format(name))
            if (q_layer_data != q_layer_data.round()).any():
                raise Exception("Quantized weights of {} layer include non-integer values".format(name))


def visualize_weights(model, save_path):
    weight_list = []
    for name,layer in list(model.named_modules()):
        if hasattr(layer, 'weight'):
            weight_list.extend(layer.weight.data.cpu().view(-1).numpy())

    plt.hist(weight_list,  bins = 'auto', align='left', color='blue', edgecolor='black')

    # Adding titles and labels
    plt.title('Weights distribution')
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    # Show the plot
    print('saved')
    plt.savefig(save_path)


#generalized hooking
def register_activation_profiling_hooks(model: nn.Module):
    model.profile_activations = True
    found_first = False
    model.input_activation_bounds = [0,0]

    def input_hook(module, input):
        if model.profile_activations:
            max = torch.max(input[0])
            min = torch.min(input[0])
            if min < model.input_activation_bounds[0]:
                model.input_activation_bounds[0] = min
            if max > model.input_activation_bounds[1]:
                model.input_activation_bounds[1] = max

    def activation_hook(layer):
        if not hasattr(layer, 'activations'):
            layer.activation_bounds = [0,0]

        def hook(module, input, output):
            # Directly append the flattened output to the list
            if model.profile_activations:
                max = torch.max(output)
                min = torch.min(output)
                if min < module.activation_bounds[0]:
                    module.activation_bounds[0] = min
                if max > module.activation_bounds[1]:
                    module.activation_bounds[1] = max
                # module.activations.append(activation)

        return hook

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            if not found_first: #pick a deit layer here and see what the input is in the input hook 
                found_first = True
                layer.register_forward_pre_hook(input_hook)
            layer.register_forward_hook(activation_hook(layer))

def clear_activations(model: nn.Module):
    model.profile_activations = False

class modelQuantized(nn.Module):
    def __init__(self, max, min, net: nn.Module):
        super(modelQuantized, self).__init__()
        self.net = net
        self.quantized_layers = []
        self.max = max
        self.min = min

        # Identify and process each quantizable layer
        #removes the wrapper model 
        for name, module in list(self.net.named_modules()):
            self.register_pre_hooks(module)
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.quantized_layers.append(module)
                # self.register_pre_hooks(module)

        self.input_activation_bounds = net.input_activation_bounds
        self.input_scale = self.quantize_initial_input(self.input_activation_bounds)
        self.setup_output_scales()

    def register_pre_hooks(self, module):
        # Define and register a forward pre-hook
        def pre_hook(layer, input):
            output_scale = getattr(layer, 'output_scale', 1)
            x = input[0]
            x = torch.clamp(torch.round(x * output_scale), min=self.min, max=self.max)
            if (x < self.min).any() or (x > self.max).any():
                raise Exception(f"Input to {layer.__class__.__name__} layer is out of bounds for an 8-bit signed integer")
            if (x != x.round()).any():
                raise Exception(f"Input to {layer.__class__.__name__} layer has non-integer values")
            return (x,) + input[1:] 

        module.register_forward_pre_hook(pre_hook)

    def setup_output_scales(self):
        # Calculate and set the output scales and quantize biases for quantizable layers
        preceding_layer_scales = []
        for layer in self.quantized_layers:
            layer.output_scale = self.quantize_activations(getattr(layer, 'activation_bounds', [0,1]), getattr(layer, 'weight.scale', 1), self.input_scale, preceding_layer_scales)
            preceding_layer_scales.append((getattr(layer, 'weight.scale', 1), layer.output_scale))

            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data = self.quantized_bias(
                    layer.bias.data,
                    getattr(layer, 'weight.scale', 1),
                    self.input_scale,
                    preceding_layer_scales
                )
                if (layer.bias.data < -2147483648).any() or (layer.bias.data > 2147483647).any():
                    raise Exception("Bias for layer {} has values which are out of bounds for a 32-bit signed integer".format(layer.__class__.__name__))
                if (layer.bias.data != layer.bias.data.round()).any():
                    raise Exception("Bias for layer {} has non-integer values".format(layer.__class__.__name__))

    def quantize_initial_input(self,bounds: np.ndarray) -> float:
        return (self.max - self.min) / (bounds[1] - bounds[0])

    def quantize_activations(self,activation_bounds, n_w: float, n_initial_input: float, ns: List[Tuple[float, float]]) -> float:
        cumulative_scale = n_initial_input * n_w
        for weight_scale, output_scale in ns:
            cumulative_scale *= output_scale * weight_scale
        return (self.max - self.min) / (activation_bounds[1] - activation_bounds[0]) / cumulative_scale

    @staticmethod
    def quantized_bias(bias: torch.Tensor, n_w: float, n_initial_input: float, ns: List[Tuple[float, float]]) -> torch.Tensor:
        cumulative_scale = n_initial_input * n_w
        for weight_scale, output_scale in ns:
            cumulative_scale *= output_scale * weight_scale
        return torch.clamp((bias * cumulative_scale).round(), min=-2147483648, max=2147483647)

    def forward(self, pixel_values, labels = None) -> torch.Tensor:
        x = pixel_values
        # Apply initial scaling
        x = torch.clamp(torch.round(x * self.input_scale), min=self.min, max=self.max)
        x = self.net(x, labels=labels)
        return x
