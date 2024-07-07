import torch

def maximise_entropy_with_zero_penalty(activations, zero_penalty=0.1):
	# This method attempts to minimise the variance in the activations;
	# there may be better methods but for now this is good enough
	# Crucically the variance is applied independently to each item in the batch
	variances = torch.var(activations, dim=1).squeeze() # (batch_size)    
	anti_sparsity = torch.mean(torch.abs(activations), dim=1).squeeze()
	return variances - anti_sparsity * zero_penalty