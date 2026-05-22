import torch
import time
from sds.hot_swap.validator import PatchValidator

def save_patch(delta_patch, filepath, metadata=None):
    """
    Saves delta_patch with a cryptographic hash for integrity.
    """
    # 1. Generate the hash before saving (The Trust Layer)
    patch_hash = PatchValidator.calculate_checksum(delta_patch)
    
    # 2. Package everything into the payload
    payload = {
        'metadata': metadata or {},
        'delta': delta_patch,
        'hash': patch_hash  # Storing the fingerprint inside the file
    }
    
    # 3. Write to .sds format
    torch.save(payload, filepath)
    print(f"Successfully saved patch to {filepath} | Hash: {patch_hash[:10]}...")

def load_patch(filepath):
    """
    Loads .sds patch and returns (delta, metadata, hash).
    """
    payload = torch.load(filepath)
    return payload['delta'], payload['metadata'], payload['hash']