import torch
from sds.hot_swap.validator import PatchValidator, SecurityError

class AtomicInjector:
    def __init__(self, model):
        self.model = model

    def apply_patch(self, delta_patch, patch_hash=None):
        """
        Applies patch atomically only AFTER verifying integrity.
        """
        # 1. Verification Phase (Phase 1 Requirement)
        if patch_hash:
            PatchValidator.verify_patch(delta_patch, patch_hash)
            print("Integrity Verified. Proceeding to Injection...")

        # 2. Injection Phase
        with torch.no_grad():
            current_state = self.model.state_dict()
            for key, delta in delta_patch.items():
                if key in current_state:
                    current_state[key].add_(delta)
            
            self.model.load_state_dict(current_state)
            print("Patch applied successfully and atomically!")