import torch

def extract_delta(model_v1: torch.nn.Module, model_v2: torch.nn.Module, threshold: float = 1e-5):
    """
    Computes the sparse delta between two PyTorch models.
    
    Args:
        model_v1: The base model (e.g., v1).
        model_v2: The updated model (e.g., v2).
        threshold: Values smaller than this will be treated as zero (Sparse Update).
        
    Returns:
        A dictionary containing only the significant weight changes.
    """
    state_v1 = model_v1.state_dict()
    state_v2 = model_v2.state_dict()
    
    delta_patch = {}
    
    # Iterate through all parameters (weights and biases)
    for key in state_v2.keys():
        if key in state_v1:
            # Calculate difference
            diff = state_v2[key] - state_v1[key]
            
            # Apply Sparse Thresholding
            # We only keep weights that have changed significantly
            mask = torch.abs(diff) > threshold
            
            # Keep only the changed values
            sparse_diff = diff * mask
            
            # Save only non-zero changes to reduce file size
            delta_patch[key] = sparse_diff
        else:
            # Handle new layers added in v2
            delta_patch[key] = state_v2[key]
            
    return delta_patch