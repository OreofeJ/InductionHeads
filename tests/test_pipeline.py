import pytest
import torch
from data.generate_sequences import generate_induction_sequences
from src.ablation import multi_head_ablation_hook

def test_sequence_generation_shape_and_target():
    """
    Verifies that generated induction sequences follow [A, B, ..., C, A, B, ...]
    with the exact target token matching the final element of the first pattern block.
    """
    seq_len = 8
    num_examples = 15
    dataset = generate_induction_sequences(
        num_examples=num_examples,
        seq_len=seq_len,
        vocab_size=50257,
        seed=77
    )
    
    assert len(dataset) == num_examples, "Dataset count does not match requested num_examples."
    
    for item in dataset:
        input_ids = item["input_ids"]
        target_id = item["target_id"]
        
        # input_ids should be pattern + pattern[:-1], so length = 2 * seq_len - 1
        assert len(input_ids) == (2 * seq_len) - 1, f"Expected length {(2 * seq_len) - 1}, got {len(input_ids)}."
        
        # The prefix of the second half must match the prefix of the first half
        first_half_prefix = input_ids[:seq_len - 1]
        second_half_prefix = input_ids[seq_len:]
        assert first_half_prefix == second_half_prefix, "Repeated token pattern mismatch."
        
        # The expected next token (target_id) must equal the last token of the first pattern block
        assert target_id == input_ids[seq_len - 1], "Target token does not match end of initial pattern."

def test_sequence_reproducibility():
    """
    Ensures identical seeds produce identical datasets.
    """
    ds_a = generate_induction_sequences(num_examples=5, seq_len=6, seed=77)
    ds_b = generate_induction_sequences(num_examples=5, seq_len=6, seed=77)
    
    assert ds_a == ds_b, "Datasets generated with the same seed (77) are not identical."

def test_multi_head_ablation_hook():
    """
    Verifies that the ablation hook zeroes out only the specified attention heads
    while leaving other heads untouched.
    """
    batch_size, seq_len, n_heads, d_head = 1, 10, 12, 64
    dummy_activations = torch.ones((batch_size, seq_len, n_heads, d_head))
    target_heads = [1, 5, 9]
    
    modified = multi_head_ablation_hook(dummy_activations.clone(), hook=None, head_indices=target_heads)
    
    for h in range(n_heads):
        if h in target_heads:
            assert torch.all(modified[:, :, h, :] == 0.0), f"Head {h} was not zeroed out."
        else:
            assert torch.all(modified[:, :, h, :] == 1.0), f"Untargeted head {h} was altered."

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
