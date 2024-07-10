import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from llm_attacks import AttackPrompt, MultiPromptAttack, PromptManager
from llm_attacks import get_embedding_matrix, get_embeddings


def token_gradients_with_arb_loss(model, tokenizer, input_ids, input_slice, loss_func):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids. (i.e. post tokenization of text)
    input_slice : slice or None
        The slice of the input sequence for which gradients need to be computed.
        Can be None, if all are desired to be modified.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """
    
    # get the embeddings for the input text
    embed_weights = model.get_input_embeddings().weight

    # create the distinct token vector over the batches
    one_hot = torch.zeros(
        input_ids.shape[0], # batch dim
        input_ids[:,input_slice].shape[1], # number of tokens to modify
        embed_weights.shape[0], # vocab size
        device=model.device,
        dtype=embed_weights.dtype
    )
    # we can now place the 1-hot in the correct word (token) position
    one_hot.scatter_(
        2, # the vocab dim
        input_ids[:,input_slice].unsqueeze(2),
        torch.ones(one_hot.shape[:-1]+(1,), device=model.device, dtype=embed_weights.dtype)
    )

    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights) # the vocab dim
    
    # now stitch it together with the rest of the embeddings
    # result is that only the input embeddings carry forth a gradient
    embeds = get_embeddings(model, input_ids).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], 
            input_embeds, 
            embeds[:,input_slice.stop:,:]
        ], 
        dim=1)
    activations = model(inputs_embeds=full_embeds)[:,-1,:] # (batch, final predictive token, d_e)
    loss = loss_func(activations) 

    # Create a dummy gradient tensor of ones with the same shape as the loss
    # This allows us to treat each batch-entry independently
    dummy_grad = torch.ones_like(loss)
    loss.backward(gradient=dummy_grad)
    gradients = one_hot.grad.clone()

    return gradients   