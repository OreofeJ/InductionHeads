import torch
from transformer_lens.model_bridge import TransformerBridge

def load_model(device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = TransformerBridge.boot_transformers("gpt2", device=device)
    model.enable_compatibility_mode()
    model.eval()
    return model
