import torch

def patch_head_activation_hook(activations, hook, head_idx, clean_cache):
    activations[:, :, head_idx, :] = clean_cache[hook.name][:, :, head_idx, :]
    return activations

def run_activation_patching(model, clean_input_ids, corrupted_input_ids, target_layer, target_head):
    _, clean_cache = model.run_with_cache(clean_input_ids)
    hook_name = f"blocks.{target_layer}.attn.hook_z"
    hook_fn = lambda activations, hook: patch_head_activation_hook(
        activations, hook, target_head, clean_cache
    )
    patched_logits = model.run_with_hooks(
        corrupted_input_ids,
        fwd_hooks=[(hook_name, hook_fn)]
    )
    return patched_logits
