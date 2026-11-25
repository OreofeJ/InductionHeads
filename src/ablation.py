import torch

def multi_head_ablation_hook(activations, hook, head_indices):
    for head_idx in head_indices:
        activations[:, :, head_idx, :] = 0.0
    return activations

def evaluate_multi_head_ablation(model, dataset, layer_head_pairs, sample_size=100, device="cuda"):
    layer_to_heads = {}
    for l, h, _ in layer_head_pairs:
        if l not in layer_to_heads:
            layer_to_heads[l] = []
        layer_to_heads[l].append(h)
        
    fwd_hooks = []
    for layer, heads in layer_to_heads.items():
        hook_name = f"blocks.{layer}.attn.hook_z"
        hook_fn = lambda activations, hook, h_list=heads: multi_head_ablation_hook(activations, hook, h_list)
        fwd_hooks.append((hook_name, hook_fn))
        
    correct_predictions = 0
    logit_differences = []
    
    with torch.no_grad():
        for item in dataset[:sample_size]:
            input_tensor = torch.tensor([item["input_ids"]], device=device)
            target_id = item["target_id"]
            
            logits = model.run_with_hooks(input_tensor, fwd_hooks=fwd_hooks)
            next_token_logits = logits[0, -1, :]
            predicted_token = next_token_logits.argmax().item()
            
            if predicted_token == target_id:
                correct_predictions += 1
                
            target_logit = next_token_logits[target_id].item()
            mask = torch.ones_like(next_token_logits, dtype=torch.bool)
            mask[target_id] = False
            max_other_logit = next_token_logits[mask].max().item()
            logit_differences.append(target_logit - max_other_logit)
            
    accuracy = correct_predictions / sample_size
    mean_logit_diff = sum(logit_differences) / len(logit_differences)
    return accuracy, mean_logit_diff
