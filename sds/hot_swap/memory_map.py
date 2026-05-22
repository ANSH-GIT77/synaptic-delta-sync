import torch

class VRAMManager:
    @staticmethod
    def get_vram_usage():
        """Returns current VRAM usage in MB."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            return f"{allocated:.2f} MB"
        return "N/A (CPU Mode)"

    @staticmethod
    def clear_cache():
        """Clears PyTorch cache after injection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("VRAM Cache cleared.")