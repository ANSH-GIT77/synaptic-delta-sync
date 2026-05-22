import torch
from sds.hot_swap.validator import PatchValidator
from sds.core.gpu_engine import GPUInjector # Import our new engine

class AtomicInjector:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        
        # If model is on GPU, initialize the specialized GPU engine
        if self.device.type == 'cuda':
            self.engine = GPUInjector(model)
        else:
            self.engine = None 

    def apply_patch(self, delta_patch, patch_hash=None):
        # 1. Validation (Same for both)
        if patch_hash:
            PatchValidator.verify_patch(delta_patch, patch_hash)

        # 2. Logic Routing
        if self.engine:
            self.engine.apply_patch_async(delta_patch)
        else:
            # Fallback to standard CPU injection logic
            with torch.no_grad():
                state = self.model.state_dict()
                for key, delta in delta_patch.items():
                    state[key].add_(delta)
                self.model.load_state_dict(state)
            print("CPU Injection complete.")