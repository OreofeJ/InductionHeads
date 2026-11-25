import random

def generate_induction_sequences(
    num_examples: int = 1000, 
    seq_len: int = 10, 
    vocab_size: int = 50257, 
    seed: int = 77
):
    """
    Generates sequences of the form: [A, B, C, ..., A, B]
    Expected next token: C
    """
    random.seed(seed)
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
