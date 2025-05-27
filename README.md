# Mamba Quantization

Trying to write a quantized version of Mamba w/ Int8 operations using a mix of PyTorch and Triton/CUDA/Thunderkittens

## Implementation

### 1. Slow Path 

Beginning by implementing the slowest path which doesn't use the Causal-Conv1D library. 

#### Things to implement:

- QuantizedLinear
    - GEMM Int8
    - Matrix Add Int8
- QuantizedConv1D
    - Int8 Convolution
- Mamba Chunk Scan