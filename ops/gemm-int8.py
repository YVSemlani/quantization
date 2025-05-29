import torch
import triton
import triton.language as tl
import time
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@triton.jit
def int8_matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride variables - how much to increment pointer when moving by 1 element
    # in each dimension. Critical for handling non-contiguous memory layouts.
    stride_am, stride_ak,  # A matrix strides (M, K dimensions)
    stride_bk, stride_bn,  # B matrix strides (K, N dimensions)  
    stride_cm, stride_cn,  # C matrix strides (M, N dimensions)
    # Meta-parameters - compile-time constants for optimal code generation
    BLOCK_SIZE_M: tl.constexpr,  # 16x16 blocks target tensor cores
    BLOCK_SIZE_N: tl.constexpr,  # Must be multiples of 16 for int8 tensor cores
    BLOCK_SIZE_K: tl.constexpr,  # K dimension blocking for memory efficiency
):

    # TO-DO: ADD QUANTIZATION MULTIPLICATIONS
    """
    Int8 Matrix Multiplication Kernel: C = A @ B
    
    Design Choices Explained:
    1. Block sizes of 16x16: Modern GPUs (A100, H100) have tensor cores optimized 
       for 16x16 int8 operations. This ensures we hit the fast tensor core path.
    2. Int8 input, Int32 accumulation: Prevents overflow during accumulation while
       maintaining precision. This is the standard tensor core int8 flow.
    3. Tiled computation: Breaks large matrices into smaller tiles that fit in
       shared memory, enabling data reuse and bandwidth optimization.
    4. 2D program grid: Each program handles one output tile, maximizing parallelism.
    
    Matrix Shapes:
    - A: (M, K) in int8
    - B: (K, N) in int8  
    - C: (M, N) in int32 (accumulator precision)
    """
    
    # CORRECTED: Get the current program's position from TRUE 2D grid
    # Each axis corresponds directly to output matrix dimensions
    pid_m = tl.program_id(axis=0)  # Row tile index (M dimension)
    pid_n = tl.program_id(axis=1)  # Column tile index (N dimension)

    # Get the starting points for the current tile
    # Ignore K b/c we iterate from 0 to K in chunks of BLOCK_SIZE_K during the multiplication loop
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    # Create lists with all m, n, and k indices we will access to create the tile
    # Indices_k doesn't use a start point b/c we always iterate from 0 to BLOCK_SIZE_K
    indices_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    indices_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    indices_k = tl.arange(0, BLOCK_SIZE_K) # Shape: (BLOCK_SIZE_K,)

    
    # Create row, column pairs 
    # Columns for A always begin as 0 -> 16 since they are A's k dimension
    # Rows for B always begin as 0 -> 16 since they are B's k dimension
    rows_a = indices_m[:, None] # Shape: (BLOCK_SIZE_M, 1)
    cols_a = indices_k[None, :] # Shape: (1, BLOCK_SIZE_K)
    rows_b = indices_k[:, None] # Shape: (BLOCK_SIZE_K, 1)
    cols_b = indices_n[None, :] # Shape: (1, BLOCK_SIZE_N)

    # Calculate base pointers for the current tiles
    a_ptrs = a_ptr + (rows_a * stride_am + cols_a * stride_ak)
    b_ptrs = b_ptr + (rows_b * stride_bk + cols_b * stride_bn)

    # Initialize accumulator for this tile
    # Use int32 accumulation to prevent overflow of int8 x int8 multiplication
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    # Multiplication Loop
    # Start with the left-most tile of A and the top-most tile of B
    # Then iterate over the K dimension in chunks of BLOCK_SIZE_K moving right in A and down in B by doing so
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Check bounds to handle matrices not perfectly divisible by block size
        # This prevents out-of-bounds memory accesses
        k_remaining = K - k * BLOCK_SIZE_K

        # k_mask is a boolean mask that is True for all indices in indices_k that are less than k_remaining
        # this removes the out of bounds indices from the tile
        k_mask =  indices_k < k_remaining
    
        # Create 2D masks to make sure we're within the bounds of A and B
        # Uses the same tricks as when we created the row, column pairs
        # rows_a[:, None] is a 2D array of shape (BLOCK_SIZE_M, 1); Essentially just stacking the booleans as such: [[True], [True], [True], ...]
        # cols_a[None, :] is a 2D array of shape (1, BLOCK_SIZE_K); Essentially just stacking the booleans as such: [[True, True, True, ...]]
        # indices_k[None, :] is a 2D array of shape (1, BLOCK_SIZE_K); Essentially just stacking the booleans as such: [[True, True, True, ...]]
        # cols_b[:, None] is a 2D array of shape (BLOCK_SIZE_N, 1); Essentially just stacking the booleans as such: [[True], [True], [True], ...]
        # The & operator is doing the element-wise boolean AND operation
        # Functionally, if the row is False or and value in the k dimension is False, the result is False
        # This makes sense as if a row is out of bounds, we don't want to load any data from it even if the columns are in bounds
        # Similarly, if a value in the k dimension is False, we don't want to load any data from it even if the rows are in bounds

        a_mask = (rows_a < M) & (indices_k[None, :] < k_remaining)
        b_mask = (indices_k[:, None] < k_remaining) & (cols_b < N)

        # Finally! We can load the tiles from host memory to device memory (I assume this is what Triton is doing here)
        # We expect the tiles to be in int8 format, so we don't convert them
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.int8)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.int8)

        # Perform the tile multiplication and accumulate
        # This is where the tensor core magic happens!
        # Triton should automatically lower this to tensor core instructions when it detects int8 inputs with appropriate block sizes
        # Add to the accumulator so we aren't overwriting the previous multiplication results
        accumulator += tl.dot(a_tile, b_tile, out_dtype=tl.int32)

        # Advance along the k dimension
        # stride_ak is how much we must increment to move one element in the k dimension of A
        # stride_bk is how much we must increment to move one element in the k dimension of B
        # We want to move BLOCK_SIZE_K elements in the k dimension thus we multiply the strides by BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Convert accumulated results to int32 for output
    # We do this b/c this kernel does not perform the quantization step
    # The quantization step will instead be performed in PyTorch
    c_tile = accumulator.to(tl.int32)
    
    # Calculate output addresses and store the result
    rows_c = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cols_c = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * rows_c[:, None] + stride_cn * cols_c[None, :]

    # Handle out of bounds indices in C
    c_mask = (rows_c[:, None] < M) & (cols_c[None, :] < N)
    tl.store(c_ptrs, c_tile, mask=c_mask)

def int8_matmul(a: torch.Tensor, b: torch.Tensor, BLOCK_SIZE_M: int = 64, BLOCK_SIZE_N: int = 64, BLOCK_SIZE_K: int = 64) -> torch.Tensor:
    """
    High-level interface for int8 matrix multiplication using Triton.
    
    Args:
        a: Input tensor A of shape (M, K) with dtype torch.int8
        b: Input tensor B of shape (K, N) with dtype torch.int8
        
    Returns:
        Output tensor C of shape (M, N) with dtype torch.int32
        
    Design Notes:
    - Returns int32 to prevent overflow during accumulation
    - Automatically handles non-contiguous tensors via stride computation
    - Grid size is calculated to cover the entire output matrix
    """
    # Validate input tensors
    assert a.dtype == torch.int8, f"Expected int8 for tensor a, got {a.dtype}"
    assert b.dtype == torch.int8, f"Expected int8 for tensor b, got {b.dtype}"
    assert a.shape[1] == b.shape[0], f"Incompatible shapes: {a.shape} @ {b.shape}"
    
    # Extract matrix dimensions
    M, K = a.shape
    K_b, N = b.shape
    
    # Create output tensor with int32 precision to prevent overflow
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)
    
    # Calculate 2D grid dimensions
    grid_m = triton.cdiv(M, BLOCK_SIZE_M)  # Number of row tiles
    grid_n = triton.cdiv(N, BLOCK_SIZE_N)  # Number of column tiles
    grid = (grid_m, grid_n)
    
    # Launch the kernel with 2D grid
    int8_matmul_kernel[grid](
        a, b, c,                    # Tensor pointers
        M, N, K,                    # Matrix dimensions
        a.stride(0), a.stride(1),   # A matrix strides
        b.stride(0), b.stride(1),   # B matrix strides  
        c.stride(0), c.stride(1),   # C matrix strides
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K # Block sizes
    )
    
    return c

ref_lib = 'cuBLAS' if torch.cuda.is_available() else 'rocBLAS'

configs = []
for fp8_inputs in [False, True]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],  # Label name for the lines
            line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": fp8_inputs},
        ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randint(-16, 16, (M, K), dtype=torch.int8, device=DEVICE)
    b = torch.randint(-16, 16, (K, N), dtype=torch.int8, device=DEVICE)

    a_fp = a.to(torch.float32)
    b_fp = b.to(torch.float32)

    quantiles = [0.5, 0.2, 0.8]
    
    # Triton's GPU timing
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a_fp, b_fp), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: int8_matmul(a, b), quantiles=quantiles)

    
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    
    return perf(ms), perf(max_ms), perf(min_ms)

def wall_clock_benchmark_with_plot():
    """Create wall clock time benchmark with custom plotting"""
    
    # Test matrix sizes (same as your existing benchmark)
    sizes = [128 * i for i in range(2, 33)]
    
    cublas_times = []
    triton_times = []
    
    print("Running wall clock benchmark...")
    
    for size in sizes:
        M, N, K = size, size, size
        print(f"Testing size {size}x{size}...")
        
        # Create test matrices
        a = torch.randint(-16, 16, (M, K), dtype=torch.int8, device=DEVICE)
        b = torch.randint(-16, 16, (K, N), dtype=torch.int8, device=DEVICE)
        a_fp = a.to(torch.float32)
        b_fp = b.to(torch.float32)
        
        # Warmup
        for _ in range(10):
            torch.matmul(a_fp, b_fp)
            int8_matmul(a, b)
        torch.cuda.synchronize()
        
        # Benchmark cuBLAS
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            torch.matmul(a_fp, b_fp)
        torch.cuda.synchronize()
        cublas_time = (time.time() - start_time) * 1000 / 100  # ms per iteration
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            int8_matmul(a, b)
        torch.cuda.synchronize()
        triton_time = (time.time() - start_time) * 1000 / 100  # ms per iteration
        
        cublas_times.append(cublas_time)
        triton_times.append(triton_time)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(sizes, cublas_times, 'b-', label='cuBLAS FP32', linewidth=2)
    plt.plot(sizes, triton_times, 'g-', label='Triton INT8', linewidth=2)
    
    plt.xlabel('Matrix Size (M=N=K)')
    plt.ylabel('Wall Clock Time (ms)')
    plt.title('Wall Clock Time Comparison: cuBLAS FP32 vs Triton INT8')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale often works better for timing plots
    
    # Save the plot
    plt.savefig('wall_clock_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\nWall Clock Time Summary:")
    print(f"Average cuBLAS time: {np.mean(cublas_times):.3f}ms")
    print(f"Average Triton time: {np.mean(triton_times):.3f}ms")
    print(f"Average speedup: {np.mean(cublas_times)/np.mean(triton_times):.2f}x")

# run benchmarks
if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True, save_path="int8_matmul_benchmark.png")
    wall_clock_benchmark_with_plot()