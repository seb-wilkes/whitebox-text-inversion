import torch

def get_random_tokens(tokenizer, batch_size, num_tokens):
    """
    Get a batch of random tokens from the tokenizer.
    """
    tokens = torch.randint(0, tokenizer.vocab_size, (batch_size, num_tokens))
    return tokens


