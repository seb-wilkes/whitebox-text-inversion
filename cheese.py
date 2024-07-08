import numpy as np 
import torch

import argparse
from src.llm_interface import LLMInterface
from src.optimization.optimizer import Optimizer
from src.optimization.loss_functions import get_loss_function

def main(args):

    loss_fn = get_loss_function(args.loss)
    optimizer = Optimizer(llm, loss_fn)
    
    result = optimizer.optimize(args.input, args.iterations)
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="LLM model to use")
    parser.add_argument("--loss", help="Loss function to use")
    parser.add_argument("--input", help="Initial input to optimize")
    parser.add_argument("--iterations", type=int, help="Number of optimization iterations")
    
    args = parser.parse_args()
    main(args)