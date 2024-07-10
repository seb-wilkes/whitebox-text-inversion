import numpy as np 
import torch

import argparse
from src.llm_interface import LLMInterface
from src.optimization.optimizer import Optimizer
from src.optimization.loss_functions import get_loss_function
from src.models import get_model_wrapper

def main(args):
    
    t_model, t_tokenizer = get_model_wrapper(args.model)
    loss_fn = get_loss_function(args.loss)
    optimizer_func = method_manager(args.optimiser)
    output_tokens, loss_history = optimizer_func(, t_model, loss_fn,\
                args.num_opt_tokens, args.iterations, args.batch_size, 
                prefix=args.prefix, **kwargs)
    
    # now we can save the output tokens and loss history
    np.save(args.save_loc + "_tokens.npy", output_tokens.cpu().numpy())
    np.save(args.save_loc + "_loss.npy", loss_history)
    
    return output_tokens, loss_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="LLM model to use")
    parser.add_argument("--layer", type=int, default=-2, help="Which layer to truncate the model at")
    parser.add_argument("--loss", help="Loss function to use")
    parser.add_argument("--optimiser", help="Optimisation routine to use (e.g. GCG)")
    parser.add_argument("--number_opt_tokens", help="Number of tokens to perform search over")
    parser.add_argument("--prefix", type=str, default=None, help="Optional: fixed prefix to use for the input")
    parser.add_argument("--batch_size", type=int, help="Batch size for optimization routine")
    parser.add_argument("--iterations", type=int, help="Number of optimization iterations")
    parser.add_argument("--save_loc", type=str, help="Where to save the output tokens and loss history")
    parser.add_argument("--run_suffix", type=str, default="", help="Optionally save the output with a suffix to the filename")
    
    args = parser.parse_args()
    main(args)