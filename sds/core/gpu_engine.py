import torch
from sds.hot_swap.memory_map import VRAMManager

class GPUInjector:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def apply_patch_async(self, delta_patch):
        """Injects patch directly into GPU memory streams."""
        if self.device.type != 'cuda':
            raise RuntimeError("Model is not on GPU. Use AtomicInjector instead.")
            
        with torch.no_grad():
            # Using streams to ensure zero-stutter injection
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                current_state = self.model.state_dict()
                for key, delta in delta_patch.items():
                    # Moving delta to GPU synchronously/asynchronously based on tensor location
                    current_state[key].add_(delta.to(self.device))
                self.model.load_state_dict(current_state)
        
        # Cleanup and Sync
        torch.cuda.synchronize()
        VRAMManager.clear_cache()
        
        print(f"GPU Injection complete. Current Usage: {VRAMManager.get_vram_usage()}")