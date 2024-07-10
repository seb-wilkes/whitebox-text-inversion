import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


IMPLEMENTED_METHODS = ['gcg']

def method_manager(method_request):
    
    if method_request.lower() not in IMPLEMENTED_METHODS:
		raise ValueError(f"Method {method_request} not implemented")
    
    if method_request.lower() == 'gcg':
        from gcg_pipeline.gcg_custom_functionality import \
            gcg_generate_adversarial_suffix_for_arb_loss
		return gcg_generate_adversarial_suffix_for_arb_loss

# , model, tokenizer, loss_func, modifiable_prompt_length, max_iters, batch_size, prefix=None, **kwargs)