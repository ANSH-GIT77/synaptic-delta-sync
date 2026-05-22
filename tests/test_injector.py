import pytest
import torch
from unittest.mock import MagicMock, patch
from sds.core.injection import AtomicInjector, SDSError

# Fixture: Yeh har test ke liye ek fresh model provide karega
@pytest.fixture
def mock_model():
    model = MagicMock()
    # Mocking model.parameters() to return an ITERATOR
    mock_param = MagicMock()
    mock_param.device = torch.device('cpu')
    
    # IMPORTANT: iter() daalna zaroori hai taaki next() kaam kare
    model.parameters.return_value = iter([mock_param]) 
    return model

def test_initialization(mock_model):
    """Check karo injector sahi se initialize ho raha hai."""
    injector = AtomicInjector(mock_model)
    assert injector.model == mock_model
    assert injector.engine is None  # CPU mode mein engine None hona chahiye

def test_apply_patch_validation_failure(mock_model):
    """Check karo ki agar validation fail ho, toh SDSError raise ho."""
    injector = AtomicInjector(mock_model)
    
    # PatchValidator.verify_patch ko mock kar rahe hain taaki wo error fake kare
    with patch('sds.core.injection.PatchValidator.verify_patch', side_effect=Exception("Invalid Hash")):
        with pytest.raises(SDSError) as excinfo:
            injector.apply_patch({"layer1": torch.tensor([0.1])}, patch_hash="wrong_hash")
        
        assert "Patch validation failed" in str(excinfo.value)

def test_cpu_injection_success(mock_model):
    """Check karo CPU injection successfully update kar raha hai."""
    # Model ka state_dict mock kar rahe hain
    mock_model.state_dict.return_value = {"layer1": torch.tensor([0.5])}
    
    injector = AtomicInjector(mock_model)
    patch_data = {"layer1": torch.tensor([0.1])}
    
    # Injection run karo
    injector.apply_patch(patch_data)
    
    # Verify: load_state_dict call hua ki nahi?
    assert mock_model.load_state_dict.called