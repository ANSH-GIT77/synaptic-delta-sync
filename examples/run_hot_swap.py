import torch
import torch.nn as nn
import copy
from sds.core.delta_engine import extract_delta
from sds.protocol.serializer import save_patch, load_patch
from sds.core.injection import AtomicInjector

# 1. Setup Models
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

model_original = SimpleModel()
model_updated_ref = copy.deepcopy(model_original)

# 2. Modify "Model Updated"
with torch.no_grad():
    model_updated_ref.fc.weight.add_(0.5) 

# 3. Full Secure Lifecycle
print("--- Step 1: Extracting & Saving Secure Patch ---")
delta = extract_delta(model_original, model_updated_ref)
save_patch(delta, "secure_patch.sds") # Now includes Hash inside the file

# 4. Hot-Swap Injection with Security Handshake
print("\n--- Step 2: Loading & Injecting with Integrity Check ---")
injector = AtomicInjector(model_original)
loaded_delta, metadata, patch_hash = load_patch("secure_patch.sds")

# Injecting with the hash verification
injector.apply_patch(loaded_delta, patch_hash=patch_hash)

# 5. Verification
print("\n--- Step 3: Final Verification ---")
diff = torch.abs(model_original.fc.weight - model_updated_ref.fc.weight).sum()
if diff < 1e-5:
    print("SUCCESS: Secure Hot-Swap Verified!")
else:
    print("FAILURE: Weights do not match.")