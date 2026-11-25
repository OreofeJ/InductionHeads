import torch

def compute_induction_scores(model, dataset, sample_size=50, device="cuda"):
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    induction_scores = torch.zeros((n_layers, n_heads), device=device)
    count = 0
    
    with torch.no_grad():
        for item in dataset[:sample_size]:
            input_tensor = torch.tensor([item["input_ids"]], device=device)
            seq_len = len(item["input_ids"])
            pattern_len = item["pattern_length"]
            
            _, cache = model.run_with_cache(input_tensor)
            
            for layer in range(n_layers):
                attn_key = f"blocks.{layer}.attn.hook_pattern"
                if attn_key in cache:
                    attn = cache[attn_key][0]
                    for h in range(n_heads):
                        head_attn = attn[h]
                        score = 0.0
                        valid_positions = 0
                        for i in range(pattern_len, seq_len):
                            target_src = i - pattern_len + 1
                            if 0 <= target_src < seq_len:
                                score += head_attn[i, target_src].item()
                                valid_positions += 1
                        if valid_positions > 0:
                            induction_scores[layer, h] += (score / valid_positions)
            count += 1
            
    induction_scores /= count
    return induction_scores.cpu().numpy()
