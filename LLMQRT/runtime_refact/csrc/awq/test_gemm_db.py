"""
Test script to compare GEMM with and without double buffering.

Usage:
    python test_gemm_db.py
"""

import torch
import time
import numpy as np

try:
    from runtime.sq_fp8_kernels import awq_gemm, awq_gemm_db
except ImportError:
    print("Error: AWQ kernels not available. Please compile first.")
    print("Make sure to include gemm_db.cu in your setup.py sources list.")
    exit(1)


def create_test_data(M, IC, OC, group_size=128, device='cuda'):
    """Create test data for AWQ GEMM."""
    x = torch.randn(M, IC, dtype=torch.float16, device=device)
    qweight = torch.randint(0, 2**31, (IC, OC // 8), dtype=torch.int32, device=device)
    scales = torch.randn(IC // group_size, OC, dtype=torch.float16, device=device).abs() * 0.1
    qzeros = torch.randint(0, 15, (IC // group_size, OC // 8), dtype=torch.int32, device=device)
    return x, qweight, scales, qzeros


def benchmark(kernel_fn, x, qweight, scales, qzeros, split_k, warmup=10, iterations=100):
    """Benchmark a kernel."""
    for _ in range(warmup):
        out = kernel_fn(x, qweight, scales, qzeros, split_k)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        out = kernel_fn(x, qweight, scales, qzeros, split_k)
    
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / iterations * 1000
    
    return out, elapsed


def test_configuration(M, IC, OC, split_k=8, group_size=128):
    """Test a specific configuration."""
    print(f"\n{'='*80}")
    print(f"Configuration: M={M}, IC={IC}, OC={OC}, split_k={split_k}, G={group_size}")
    print(f"{'='*80}")
    
    # Create test data
    x, qweight, scales, qzeros = create_test_data(M, IC, OC, group_size)
    
    # Test original GEMM
    print("\n[1] Testing Original GEMM...")
    try:
        out_orig, time_orig = benchmark(awq_gemm, x, qweight, scales, qzeros, split_k)
        print(f"    Time: {time_orig:.3f} ms")
        print(f"    Output shape: {out_orig.shape}")
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None
    
    # Test double-buffered GEMM
    print("\n[2] Testing Double-Buffered GEMM...")
    try:
        out_db, time_db = benchmark(awq_gemm_db, x, qweight, scales, qzeros, split_k)
        print(f"    Time: {time_db:.3f} ms")
        print(f"    Output shape: {out_db.shape}")
        
        speedup = time_orig / time_db
        print(f"    Speedup: {speedup:.2f}x")
        
        # Verify correctness
        print("\n[3] Verifying Correctness...")
        if torch.allclose(out_orig, out_db, rtol=1e-2, atol=1e-2):
            print("    ✅ Results match!")
        else:
            diff = (out_orig - out_db).abs()
            print(f"    ⚠️  Max diff: {diff.max().item():.6f}")
            print(f"    ⚠️  Mean diff: {diff.mean().item():.6f}")
            print(f"    ⚠️  Relative error: {(diff / (out_orig.abs() + 1e-6)).mean().item():.6f}")
        
        return {
            'M': M, 'IC': IC, 'OC': OC, 'split_k': split_k,
            'time_orig': time_orig,
            'time_db': time_db,
            'speedup': speedup
        }
    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("\n" + "="*80)
    print("AWQ GEMM Double Buffering Performance Test")
    print("="*80)
    
    # Test configurations
    configs = [
        # (M, IC, OC, split_k)
        (16, 4096, 4096, 8),      # Standard LLM layer
        (32, 4096, 4096, 8),      # Larger batch
        (16, 4096, 11008, 8),     # FFN intermediate
        (16, 11008, 4096, 8),     # FFN output
        (64, 4096, 4096, 8),      # Very large batch
        (16, 2048, 2048, 4),      # Smaller model
        (16, 8192, 8192, 8),      # Larger model
    ]
    
    results = []
    for M, IC, OC, split_k in configs:
        result = test_configuration(M, IC, OC, split_k)
        if result:
            results.append(result)
    
    # Summary
    if results:
        print("\n" + "="*80)
        print("Performance Summary")
        print("="*80)
        print(f"{'Config':<30} {'Original':<12} {'Double-Buf':<12} {'Speedup':<10}")
        print("-"*80)
        
        for r in results:
            config_str = f"M={r['M']}, IC={r['IC']}, OC={r['OC']}"
            print(f"{config_str:<30} {r['time_orig']:>8.3f} ms  {r['time_db']:>8.3f} ms  {r['speedup']:>6.2f}x")
        
        avg_speedup = np.mean([r['speedup'] for r in results])
        print("-"*80)
        print(f"Average Speedup: {avg_speedup:.2f}x")
        print("="*80)
        
        # Memory analysis
        print("\n" + "="*80)
        print("Memory Analysis (per block)")
        print("="*80)
        print("Original GEMM:")
        print("  - A_shared: 16 × 40 = 640 half = 1.25 KB")
        print("  - B_shared: 32 × 136 = 4352 half = 8.5 KB")
        print("  - Total: 9.75 KB")
        print("\nDouble-Buffered GEMM:")
        print("  - A_shared[2]: 2 × 1.25 KB = 2.5 KB")
        print("  - B_shared[2]: 2 × 8.5 KB = 17.0 KB")
        print("  - Total: 19.5 KB")
        print("\nOccupancy Impact:")
        print("  - SM shared memory: 164 KB (A100) or 227 KB (H100)")
        print("  - Original: can run ~16 blocks/SM")
        print("  - Double-buffered: can run ~8 blocks/SM")
        print("  - Trade-off: Lower occupancy but better pipeline efficiency")
        print("="*80)


if __name__ == "__main__":
    main()

