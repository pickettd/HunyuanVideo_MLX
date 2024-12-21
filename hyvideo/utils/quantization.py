import mlx.core as mx
import numpy as np
from typing import Dict, Any, Tuple

def quantize_weight(weight: mx.array, bits: int = 4) -> Tuple[mx.array, Dict[str, Any]]:
    """Quantize weights to n-bits.
    
    Args:
        weight: Input weight tensor
        bits: Number of bits for quantization (default: 4)
        
    Returns:
        Tuple of (quantized_weight, quantization_params)
    """
    # Get weight range
    w_min = weight.min()
    w_max = weight.max()
    
    # Compute scale and zero point
    qmax = 2**bits - 1
    scale = (w_max - w_min) / qmax
    zero_point = -w_min / scale
    
    # Quantize
    quantized = mx.clip(
        mx.round((weight / scale) + zero_point),
        0,
        qmax
    )
    
    # Store quantization params
    params = {
        "scale": scale,
        "zero_point": zero_point,
        "bits": bits,
        "w_min": w_min,
        "w_max": w_max
    }
    
    return quantized, params

def dequantize_weight(quantized: mx.array, params: Dict[str, Any]) -> mx.array:
    """Dequantize weights using stored parameters.
    
    Args:
        quantized: Quantized weight tensor
        params: Quantization parameters from quantize_weight
        
    Returns:
        Dequantized weight tensor
    """
    return (quantized - params["zero_point"]) * params["scale"]

class QuantizedLinear(mx.nn.Linear):
    """Quantized linear layer implementation."""
    
    def __init__(self, weight, bias=None, bits=4):
        super().__init__(weight.shape[1], weight.shape[0])
        # Quantize weights
        self.quantized_weight, self.quant_params = quantize_weight(weight, bits)
        self.bias = bias
        self.bits = bits
        
    def __call__(self, x):
        # Dequantize weights for computation
        weight = dequantize_weight(self.quantized_weight, self.quant_params)
        out = x @ weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

def quantize_model_weights(model: Any, bits: int = 4) -> None:
    """Quantize all linear layer weights in a model.
    
    Args:
        model: MLX model to quantize
        bits: Number of bits for quantization
    """
    for name, module in model.named_modules():
        if isinstance(module, mx.nn.Linear):
            # Replace linear layer with quantized version
            setattr(
                model,
                name,
                QuantizedLinear(
                    module.weight,
                    module.bias,
                    bits=bits
                )
            )
