from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn

import argparse

import json
from tqdm import tqdm
import os

LLAMA_PATH = '../../llama2/'
DEVICE = 'cuda:5'

def get_model_output(input, model, tokenizer):
    input_ids = tokenizer.encode(input, return_tensors='pt').to(model.device)

    output = model.generate(input_ids, max_length=200)

    # breakpoint()

    return tokenizer.decode(output[0], skip_special_tokens=True) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input', type=str, default='The meaning of life is')
    parser.add_argument('--model', type=str, default='Llama-2-7b-hf')
    parser.add_argument('--system_prompt', type=str, default='You are a helpful AI assistant.')
    parser.add_argument('--coding_prompt', type=str, default='')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--input_file_path', type=str, default='')
    # parser.add_argument('--output_file_path', type=str, default='output.json')
    args = parser.parse_args()