import torch
import torch.nn as nn
import copy
from sds.core.delta_engine import extract_delta
from sds.protocol.serializer import save_patch  # <--- Ye line add ki humne

# 1. Setup Models
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

model_v1 = SimpleModel()
model_v2 = copy.deepcopy(model_v1)

with torch.no_grad():
    model_v2.fc.weight.add_(0.5) 
    model_v2.fc.bias.add_(1e-7) 

# 2. Extract Delta
print("Extracting Delta...")
delta = extract_delta(model_v1, model_v2, threshold=1e-5)

# 3. Save to .sds file (Humara Naya Step!)
print("Saving patch to .sds file...")
metadata = {"version": "1.0", "description": "Testing SDS format"}
save_patch(delta, "test_patch.sds", metadata=metadata)

print("Done! Check your folder for test_patch.sds")