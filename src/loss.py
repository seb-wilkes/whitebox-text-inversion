import torch


IMPLEMENTED_METHODS = ['max_layer_entropy']

def get_loss_function(loss_request):    
    if loss_request.lower() not in IMPLEMENTED_METHODS:
		raise NotImplementedError(f"Loss approach '{method_request}' not implemented")
	
 	if method == 'max_layer_entropy':
		return return_maximise_entropy_with_zero_penalty(**kwargs)

def maximise_entropy_with_zero_penalty(activations, zero_penalty=0.1):
		# This method attempts to minimise the variance in the activations;
		# there may be better methods but for now this is good enough
		# Crucically the variance is applied independently to each item in the batch
		variances = torch.var(activations, dim=1).squeeze() # (batch_size)    
		anti_sparsity = torch.mean(torch.abs(activations), dim=1).squeeze()
		return variances - anti_sparsity * zero_penalty

def return_maximise_entropy_with_zero_penalty(zero_penalty=0.1):
			_loss_wraper = lambda activations: maximise_entropy_with_zero_penalty(activations, zero_penalty=zero_penalty)
			return _loss_wraper