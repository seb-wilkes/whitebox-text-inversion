from typing import Callable, Optional, Union
import torch
import numpy as np
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedModel
from tqdm import tqdm
from unmodified_from_gcg import *


def gcg_generate_adversarial_suffix_for_arb_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    loss_func: Callable[[Tensor], Tensor],
    num_opt_tokens: int,
    num_steps: int,
    batch_size: int,
    b1: int,
    b2: int,
    topk: int,
    allow_non_ascii: bool,
    prefix: Optional[str] = None,
    plotting: bool = False,
    output_strings: bool = False
) -> Union[Tuple[List[str], List[float]], Tuple[Tensor, List[float]]]:
    """
    Generate adversarial suffixes using arbitrary loss function.

    Args:
        model (PreTrainedModel): The language model.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        loss_func (Callable): The loss function to optimize.
        num_opt_tokens (int): Number of tokens to optimize.
        num_steps (int): Number of optimization steps.
        batch_size (int): Here, this means the MAXIMUM number of threads being run on the GPU
        b1 (int): Number of distinct prompts to produce.
        b2 (int): Number of candidate solutions per prompt.
        topk (int): Top-k value for sampling.
        allow_non_ascii (bool): Whether to allow non-ASCII tokens.
        prefix (Optional[str]): Prefix for the prompts.
        plotting (bool): Whether to plot the loss.
        output_strings (bool): Whether to output strings instead of tokens.

    Returns:
        Union[Tuple[List[str], List[float]], Tuple[Tensor, List[float]]]: 
            If output_strings is True:
                - A tuple containing:
                    - A list of decoded strings (the optimized prompts)
                    - A list of float values representing the loss history
            If output_strings is False:
                - A tuple containing:
                    - A tensor of token IDs (the optimized prompts)
                    - A list of float values representing the loss history
    """
    assert b1 * b2 <= batch_size, "The combined b1*b2 values are larger than the specified batch size"
    assert topk <= b2, "topk needs to be smaller than the candidate batch size"
    
    not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)
    
    initial_tokens = torch.randint(0, tokenizer.vocab_size, (b1, num_opt_tokens)) # b1 x random initial conditions    
    prefix_tokens = tokenizer.encode(prefix or "", return_tensors='pt').squeeze(0).repeat(b1, 1)
    prompt_tokens = torch.cat([prefix_tokens, initial_tokens], dim=1)
    p_offset = prefix_tokens.shape[1]
    control_slice = slice(p_offset, prompt_tokens.shape[1])
    loss_history = np.zeros((b1, num_steps))
    

    for i in tqdm(range(num_steps)):
        input_ids = prompt_tokens.clone().to(model.device)
        coordinate_grad = token_gradients_with_arb_loss(model, tokenizer, input_ids, control_slice, loss_func)
        
        with torch.no_grad():
            adv_suffix_tokens = input_ids[:, control_slice].to(model.device)
            candidate_token_sequences = []
            best_in_batch_sequences = []
            batch_losses = [] if plotting else None

            for j, batch_entry in enumerate(adv_suffix_tokens):
                b2_suggestions = sample_control(batch_entry, coordinate_grad[j], b2, topk=topk, not_allowed_tokens=not_allowed_tokens)
                candidate_token_sequences.append(light_token_filtering_checks(b2_suggestions, filter_cand=True))

            full_sequence_batch = torch.cat(candidate_token_sequences, dim=0).to(model.device)
            logits, ids = get_activations(model=model, input_ids=input_ids, control_slice=control_slice, b1=b1, b2=b2,
                                          test_controls=full_sequence_batch, return_ids=True)
            losses = loss_func(logits)

            for j in range(b1):
                minibatch_slice = slice(j*b2, (j+1)*b2)
                mini_batch_argbest = losses[minibatch_slice].argmin()
                loss_history[j,i] = losses[mini_batch_argbest].cpu().numpy()
                best_in_batch_sequences.append(candidate_token_sequences[j][mini_batch_argbest])

            prompt_tokens = torch.cat([prefix_tokens, torch.stack(best_in_batch_sequences, dim=0).cpu()], dim=1)

        if plotting:
            plotlosses.update({'Loss': np.mean(batch_losses[:,i])})
            plotlosses.send()

    if output_strings:
        return tokenizer.decode(prompt_tokens), loss_history
    else:
        return prompt_tokens, loss_history


def token_gradients_with_arb_loss(model, tokenizer, input_ids, input_slice, loss_func):

    """
    Computes gradients of the (arbitary) loss with respect to the coordinates.
    Modified from the original GCG code that finds the gradient to accomodate 
    these new requirements.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids. (i.e. post tokenization of text)
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
        Can be for the whole input sequence.
    loss_func : func
        The desired loss function that the user specifies, taking in the model
        layer's activations.

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

def light_token_filtering_checks(control_cand, filter_cand=True, curr_control=None, max_attempts=5):
    ''' 
    Modified from get_filtered_cands(...) in the original GCG code. 
    Here the main difference is this only operates with 
    tokens rather than en/decoding strings.
    '''
    cands, count = [], 0

    for i in range(control_cand.shape[0]):
        candidate = control_cand[i]
        if filter_cand:
            if curr_control is None:
                cands.append(candidate)
            elif (not torch.equal(candidate, curr_control) and len(candidate) == len(curr_control)):
                cands.append(candidate)
            else:
                count += 1 # original authors had this here but doens't  do anything ...
        else:
            cands.append(candidate)
    
    if filter_cand and len(cands) == 0: # honestly, an inelegant "solution" ... but I think this should come up rarely
        print("All candidates were filtered out. Resampling...")
        if max_attempts > 0:
            # Resample a new control_cand
            new_control_cand = torch.randint(0, control_cand.max().item() + 1, control_cand.shape, device=control_cand.device)
            return get_filtered_cands(new_control_cand, filter_cand, curr_control, max_attempts-1)
        else:
            print("Max resampling attempts reached. Adding original candidate.")
            cands.append(control_cand[0])
    
    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands)) # honestly, an inelegant "solution" ... but I think this should come up rarely
    
    return torch.stack(cands)

def get_activations(*, model, input_ids, control_slice, b1, b2, test_controls=None, return_ids=False):
    '''  
    This function modified GCG's original get_logits, leveraging the fact we 
    can directly access the logits. 
    It returns an object that is dims -> [b1*b2, 1, layer_width]
    which can be further processed downstream
    '''    
    max_len = control_slice.stop - control_slice.start
    if test_controls.shape[1] != max_len:
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {max_len}), " 
            f"got {test_controls.shape}"
        ))

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_controls.shape[0], 1).to(model.device)
    ids = torch.scatter(
        input_ids.repeat_interleave(b2, 0).to(model.device),
        1,
        locs,
        test_controls
    )

    if return_ids:
        logits = forward_wrapper(model=model, input_ids=ids, attention_mask=None, batch_size=b1*b2)[:,-1,:] # corresponding to final token activations
        return logits, ids
    else:
        logits = forward_wrapper(model=model, input_ids=ids, attention_mask=None, batch_size=b1*b2)[:,-1,:] # corresponding to final token activations
        del ids ; gc.collect()
        return logits

def forward_wrapper(*, model, input_ids, attention_mask=None, batch_size):
    ''' 
    Mildly different from the original forward function in GCG, 
    accomodating that the output of a model already are the logits.
    '''
    logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        batch_input_ids = input_ids[i:i+batch_size]
        batch_attention_mask = attention_mask[i:i+batch_size] if attention_mask is not None else None

        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask))

        gc.collect()

    del batch_input_ids, batch_attention_mask
    
    return torch.cat(logits, dim=0) 

    