import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import socket
import time

def init_distributed(rank, world_size, port):
    """Initialize distributed training with better error handling."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    
    # Add timeout for initialization
    print(f"Process {rank}: Starting initialization (port={port})")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Use a timeout for NCCL initialization
            dist.init_process_group(
                "nccl", 
                rank=rank, 
                world_size=world_size,
                timeout=torch.distributed.Store.TIMEOUT_DEFAULT
            )
            torch.cuda.set_device(rank)
            print(f"Process {rank}: Successfully initialized")
            return True
        except Exception as e:
            print(f"Process {rank}: Initialization failed (attempt {attempt+1}/{max_retries}): {str(e)}")
            time.sleep(2)
    
    print(f"Process {rank}: Failed to initialize after {max_retries} attempts")
    return False

def find_free_port():
    """Find a free port that's less likely to be taken immediately."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    # Sleep briefly to reduce chance of port collision
    time.sleep(0.1)
    return port

def test_worker(rank, world_size, port):
    """Simple worker function to test if NCCL initialization works."""
    success = init_distributed(rank, world_size, port)
    if success:
        # Create a simple tensor and all-reduce
        tensor = torch.ones(1).to(rank)
        dist.all_reduce(tensor)
        print(f"Process {rank}: All-reduce result = {tensor.item()}")
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

def run_distributed_test():
    """Run a basic distributed test to verify connectivity."""
    world_size = torch.cuda.device_count()
    port = find_free_port()
    print(f"Testing distributed setup with {world_size} GPUs on port {port}")
    mp.spawn(test_worker, nprocs=world_size, args=(world_size, port))

if __name__ == "__main__":
    run_distributed_test()
