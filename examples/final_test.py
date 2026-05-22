import torch
import torch.nn as nn
from sds.core.delta_engine import extract_delta
from sds.core.injection import AtomicInjector
from sds.protocol.serializer import save_patch, load_patch

def run_integration_test():
    print("--- Starting Enterprise Integration Test ---")
    
    # 1. Setup Model
    model = nn.Linear(10, 10)
    # Move to GPU if available to test our new GPUInjector logic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Testing on device: {device}")

    # 2. Extract Delta
    ref_model = nn.Linear(10, 10).to(device)
    ref_model.weight.data.add_(0.2)
    delta = extract_delta(model, ref_model)
    
    # 3. Save Patch
    save_patch(delta, "final_test.sds")
    
    # 4. Inject using Smart Engine
    injector = AtomicInjector(model)
    loaded_delta, _, patch_hash = load_patch("final_test.sds")
    
    print("Executing Injection...")
    injector.apply_patch(loaded_delta, patch_hash=patch_hash)
    
    # 5. Verify Result
    if torch.allclose(model.weight, ref_model.weight, atol=1e-5):
        print("SUCCESS: Engine Routing & Injection Verified!")
    else:
        print("FAILURE: Weights Mismatch.")

if __name__ == "__main__":
    run_integration_test()