import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

implemented_methods = ['gcg']

def method_manager(method_request, model, tokenizer, loss_func, modifiable_prompt_length, max_iters, batch_size, prefix=None, **kwargs):
    
    if method_request.lower() not in implemented_methods:
        import gcg_custom_functionality
		raise ValueError(f"Method {method_request} not implemented")
    
    
    if method_request.lower() == 'gcg':	
		return gcg_generate_adversarial_suffix_for_arb_loss(model, \
      			tokenizer, loss_func, modifiable_prompt_length, max_iters, batch_size, prefix, **kwargs)