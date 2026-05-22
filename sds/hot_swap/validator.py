import hashlib
import torch

class SecurityError(Exception):
    """Custom exception raised when patch integrity fails."""
    pass

class PatchValidator:
    @staticmethod
    def calculate_checksum(delta_patch):
        """Generates a SHA-256 fingerprint for the current patch."""
        # We create a deterministic string representation of the tensors
        # sum().item() is a quick way to get a unique scalar fingerprint per layer
        content = "".join([f"{key}:{t.sum().item():.6f}" for key, t in delta_patch.items()])
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def verify_patch(delta_patch, expected_hash):
        """Verifies if the patch has been tampered with before injection."""
        actual_hash = PatchValidator.calculate_checksum(delta_patch)
        
        if actual_hash != expected_hash:
            raise SecurityError(
                f"CRITICAL: Integrity Mismatch! \n"
                f"Expected: {expected_hash} \n"
                f"Actual: {actual_hash} \n"
                "Injection aborted to prevent model corruption."
            )
        return True