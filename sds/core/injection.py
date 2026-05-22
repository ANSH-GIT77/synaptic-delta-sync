import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from sds.hot_swap.validator import PatchValidator
from sds.core.gpu_engine import GPUInjector

# 1. Custom Exception class - International standard for libraries
class SDSError(Exception):
    """Base exception for Synaptic-Delta-Sync errors."""
    pass

class AtomicInjector:
    """
    Handles atomic weight updates for neural networks with GPU/CPU support.
    
    Attributes:
        model (nn.Module): The target model to update.
        engine (Optional[GPUInjector]): Engine used for asynchronous GPU injection.
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize the injector.

        Args:
            model (nn.Module): The PyTorch model to patch.
        """
        self.model = model
        self.device = next(iter(model.parameters())).device
        
        # Using type hinting to make it clear we expect an optional engine
        self.engine: Optional[GPUInjector] = None
        
        if self.device.type == 'cuda':
            self.engine = GPUInjector(model)

    def apply_patch(self, delta_patch: Dict[str, torch.Tensor], patch_hash: Optional[str] = None) -> None:
        """
        Applies a delta patch to the model weights atomically.

        Args:
            delta_patch (Dict[str, torch.Tensor]): Dictionary mapping param names to delta tensors.
            patch_hash (Optional[str]): SHA-256 hash for integrity validation.

        Raises:
            SDSError: If patch validation fails.
        """
        # 1. Validation Logic
        if patch_hash:
            try:
                PatchValidator.verify_patch(delta_patch, patch_hash)
            except Exception as e:
                # Wrapping raw errors in our Custom Exception
                raise SDSError(f"Patch validation failed: {str(e)}")

        # 2. Logic Routing
        if self.engine:
            self.engine.apply_patch_async(delta_patch)
        else:
            # CPU Fallback with standard safety checks
            try:
                with torch.no_grad():
                    state = self.model.state_dict()
                    for key, delta in delta_patch.items():
                        if key in state:
                            state[key].add_(delta)
                    # NOTE: We update in-place where possible, but state_dict approach is safer for now
                    self.model.load_state_dict(state)
            except Exception as e:
                raise SDSError(f"CPU injection failed: {str(e)}")
            
            print("CPU Injection complete.")