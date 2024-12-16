import torch
import time
import psutil
import numpy as np
from pathlib import Path
from loguru import logger

class Benchmark:
    def __init__(self):
        self.device = torch.device("mps")
        self.results = {}
        
    def _measure_memory(self):
        """Measure current memory usage"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024 ** 3),
            'used': (memory.total - memory.available) / (1024 ** 3),
            'percent': memory.percent
        }
    
    def test_tensor_operations(self, size=(1, 16, 129, 68, 120)):
        """Test basic tensor operations with typical video dimensions"""
        print("\n=== Testing Tensor Operations ===")
        try:
            # Create test tensor
            start_time = time.time()
            x = torch.randn(size, device=self.device)
            creation_time = time.time() - start_time
            
            # Test operations
            start_time = time.time()
            y = torch.nn.functional.interpolate(x.view(-1, *x.shape[2:]), 
                                              scale_factor=2, 
                                              mode='bilinear')
            y = y.view(x.shape[0], x.shape[1], -1, *y.shape[2:])
            operation_time = time.time() - start_time
            
            memory = self._measure_memory()
            
            self.results['tensor_ops'] = {
                'creation_time': creation_time,
                'operation_time': operation_time,
                'memory_usage': memory
            }
            
            print(f"✓ Tensor creation: {creation_time:.2f}s")
            print(f"✓ Operation time: {operation_time:.2f}s")
            print(f"✓ Memory usage: {memory['used']:.1f}GB ({memory['percent']}%)")
            
        except Exception as e:
            print(f"✗ Error during tensor operations: {str(e)}")
            self.results['tensor_ops'] = {'error': str(e)}
    
    def test_memory_bandwidth(self, size_gb=1):
        """Test memory bandwidth"""
        print("\n=== Testing Memory Bandwidth ===")
        try:
            # Create large tensor
            size = int(size_gb * 1024 * 1024 * 1024 / 4)  # Convert GB to float32 elements
            x = torch.randn((size,), device=self.device, dtype=torch.float32)
            
            # Measure read bandwidth
            start_time = time.time()
            y = x + 1
            torch.mps.synchronize()
            read_time = time.time() - start_time
            read_bandwidth = size_gb / read_time
            
            # Measure write bandwidth
            start_time = time.time()
            x.copy_(y)
            torch.mps.synchronize()
            write_time = time.time() - start_time
            write_bandwidth = size_gb / write_time
            
            self.results['memory_bandwidth'] = {
                'read_bandwidth': read_bandwidth,
                'write_bandwidth': write_bandwidth
            }
            
            print(f"✓ Read bandwidth: {read_bandwidth:.1f} GB/s")
            print(f"✓ Write bandwidth: {write_bandwidth:.1f} GB/s")
            
        except Exception as e:
            print(f"✗ Error during memory bandwidth test: {str(e)}")
            self.results['memory_bandwidth'] = {'error': str(e)}
    
    def test_vae_performance(self, batch_size=1, video_length=129):
        """Test VAE encoding/decoding performance"""
        print("\n=== Testing VAE Performance ===")
        try:
            # Simulate VAE input dimensions
            x = torch.randn((batch_size, 3, video_length, 544, 960), 
                          device=self.device)
            
            # Test encoding
            start_time = time.time()
            y = torch.nn.functional.interpolate(x.view(-1, *x.shape[2:]), 
                                              scale_factor=0.125, 
                                              mode='bilinear')
            y = y.view(x.shape[0], x.shape[1], -1, *y.shape[2:])
            encode_time = time.time() - start_time
            
            # Test decoding
            start_time = time.time()
            z = torch.nn.functional.interpolate(y.view(-1, *y.shape[2:]), 
                                              scale_factor=8, 
                                              mode='bilinear')
            z = z.view(y.shape[0], y.shape[1], -1, *z.shape[2:])
            decode_time = time.time() - start_time
            
            self.results['vae_performance'] = {
                'encode_time': encode_time,
                'decode_time': decode_time
            }
            
            print(f"✓ Encode time: {encode_time:.2f}s")
            print(f"✓ Decode time: {decode_time:.2f}s")
            
        except Exception as e:
            print(f"✗ Error during VAE performance test: {str(e)}")
            self.results['vae_performance'] = {'error': str(e)}
    
    def estimate_performance(self):
        """Estimate overall performance and make recommendations"""
        print("\n=== Performance Estimation ===")
        
        try:
            # Calculate performance score
            tensor_score = 1.0 / self.results['tensor_ops']['operation_time']
            memory_score = (self.results['memory_bandwidth']['read_bandwidth'] + 
                          self.results['memory_bandwidth']['write_bandwidth']) / 2
            vae_score = 1.0 / (self.results['vae_performance']['encode_time'] + 
                              self.results['vae_performance']['decode_time'])
            
            # Normalize scores
            total_score = (tensor_score + memory_score + vae_score) / 3
            
            print("\nRecommended Settings:")
            
            if total_score > 0.8:
                print("✓ High Performance - Recommended settings:")
                print("  - Resolution: 720x1280")
                print("  - Video length: 129 frames")
                print("  - Batch size: 1")
                print("  - MMGP: Optional")
            elif total_score > 0.5:
                print("! Medium Performance - Recommended settings:")
                print("  - Resolution: 544x960")
                print("  - Video length: 129 frames")
                print("  - Batch size: 1")
                print("  - MMGP: Recommended")
            else:
                print("! Lower Performance - Recommended settings:")
                print("  - Resolution: 544x960")
                print("  - Video length: 65 frames")
                print("  - Batch size: 1")
                print("  - MMGP: Required")
            
            print("\nOptimization Tips:")
            print(f"- Set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7")
            print(f"- Close other applications during generation")
            print(f"- Monitor system resources with monitor_resources.py")
            
        except Exception as e:
            print(f"✗ Error during performance estimation: {str(e)}")

def main():
    print("=== HunyuanVideo Performance Benchmark ===")
    
    benchmark = Benchmark()
    benchmark.test_tensor_operations()
    benchmark.test_memory_bandwidth()
    benchmark.test_vae_performance()
    benchmark.estimate_performance()
    
    print("\n=== Benchmark Complete ===")
    print("Use these results to optimize your configuration in configs/mmgp_example.json")

if __name__ == "__main__":
    main()
