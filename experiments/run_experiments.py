import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens.model_bridge import TransformerBridge

# --- 1. CONFIGURATION & SEEDING ---
SEED = 77
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device} (Seed: {SEED})")

# --- 2. MODEL LOADING ---
print("Loading GPT-2 Small via TransformerBridge...")
model = TransformerBridge.boot_transformers("gpt2", device=device)
model.enable_compatibility_mode()
model.eval()

# --- 3. DATASET GENERATION ---
def generate_induction_sequences(num_examples=100, seq_len=10, vocab_size=50257):
    dataset = []
    for _ in range(num_examples):
        pattern = [random.randint(100, vocab_size - 1) for _ in range(seq_len)]
        input_ids = pattern + pattern[:-1]
        target_id = pattern[-1]
        dataset.append({
            "input_ids": input_ids,
            "target_id": target_id,
            "pattern_length": seq_len
        })
    return dataset

dataset = generate_induction_sequences(num_examples=100)
print(f"Generated {len(dataset)} synthetic induction sequences.")

# --- 4. CIRCUIT DETECTION (Induction Scores) ---
def compute_induction_scores(model, dataset, sample_size=50):
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

scores = compute_induction_scores(model, dataset, sample_size=50)

# Extract top 5 heads
flat_indices = np.argsort(scores.flatten())[::-1]
top_heads = []
for idx in flat_indices[:5]:
    l = idx // model.cfg.n_heads
    h = idx % model.cfg.n_heads
    top_heads.append((l, h, scores[l, h]))

print("\nTop 5 Detected Induction Heads:")
for l, h, score in top_heads:
    print(f"  -> Layer {l}, Head {h} (Score: {score:.4f})")

# --- 5. MULTI-HEAD ABLATION EXPERIMENT ---
def multi_head_ablation_hook(activations, hook, head_indices):
    for head_idx in head_indices:
        activations[:, :, head_idx, :] = 0.0
    return activations

def evaluate_ablation(model, dataset, layer_head_pairs, sample_size=50):
    layer_to_heads = {}
    for l, h in layer_head_pairs:
        if l not in layer_to_heads:
            layer_to_heads[l] = []
        layer_to_heads[l].append(h)
        
    fwd_hooks = []
    for layer, heads in layer_to_heads.items():
        hook_name = f"blocks.{layer}.attn.hook_z"
        hook_fn = lambda activations, hook, h_list=heads: multi_head_ablation_hook(activations, hook, h_list)
        fwd_hooks.append((hook_name, hook_fn))
        
    correct = 0
    logit_diffs = []
    
    with torch.no_grad():
        for item in dataset[:sample_size]:
            input_tensor = torch.tensor([item["input_ids"]], device=device)
            target_id = item["target_id"]
            
            logits = model.run_with_hooks(input_tensor, fwd_hooks=fwd_hooks)
            next_token_logits = logits[0, -1, :]
            pred = next_token_logits.argmax().item()
            
            if pred == target_id:
                correct += 1
                
            target_logit = next_token_logits[target_id].item()
            mask = torch.ones_like(next_token_logits, dtype=torch.bool)
            mask[target_id] = False
            max_other = next_token_logits[mask].max().item()
            logit_diffs.append(target_logit - max_other)
            
    return correct / sample_size, sum(logit_diffs) / len(logit_diffs)

top_5_pairs = [(l, h) for l, h, _ in top_heads]
print("\nEvaluating Top 5 Multi-Head Ablation...")
acc, diff = evaluate_ablation(model, dataset, top_5_pairs, sample_size=50)
print(f"  -> Accuracy: {acc * 100:.2f}%")
print(f"  -> Mean Logit Diff: {diff:.4f}")

# Negative Control (Random Heads using Seed 77)
rng = np.random.default_rng(SEED)
control_pairs = []
for _ in range(5):
    control_pairs.append((rng.integers(0, model.cfg.n_layers), rng.integers(0, model.cfg.n_heads)))

print("\nEvaluating Negative Control (5 Random Heads)...")
ctrl_acc, ctrl_diff = evaluate_ablation(model, dataset, control_pairs, sample_size=50)
print(f"  -> Control Accuracy: {ctrl_acc * 100:.2f}%")
print(f"  -> Control Mean Logit Diff: {ctrl_diff:.4f}")

# --- 6. DISPLAY PLOT (No Saving) ---
print("\nGenerating visual display...")
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
conditions = ["Top 5 Induction Ablation", "Control (Random Heads)"]
accuracies = [acc * 100, ctrl_acc * 100]
plt.bar(conditions, accuracies, color=["#c44e52", "#4c72b0"], alpha=0.85, width=0.4)
plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
plt.title("Ablation Performance Comparison (Seed 77)", fontsize=13, fontweight="bold")
plt.ylim(0, 105)
plt.tight_layout()
plt.show()
