import got_utils

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn

import argparse

import json
from tqdm import tqdm
import os

import numpy as np

LLAMA_PATH = '../../llama2/'
DEVICE = 'cuda:5'
    

def get_model_activations(input, model, tokenizer):
    inputs = tokenizer(input, return_tensors='pt')
    outputs = model(**inputs, output_hidden_states=True)
    activations = [x[:, -1, :] for x in outputs.hidden_states]
    return torch.cat(activations).detach().cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input', type=str, default='The meaning of life is')
    parser.add_argument('--model', type=str, default='Llama-2-13b-hf')
    parser.add_argument('--coding_prompt', type=str, default='factual')
    parser.add_argument('--input_file_name', type=str, default='')
    parser.add_argument('--dataset', type=str, default='easy_100')
    args = parser.parse_args()

    input_file_path = os.path.join('datasets', args.coding_prompt, args.dataset, args.input_file_name)

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH + args.model,
        device_map= 'auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH + args.model)

    with open(input_file_path, 'r') as f:
        data = json.load(f)

    activations = np.stack([get_model_activations(d['query'], model, tokenizer) for d in data])

    output_file_name = f"activations_{args.model.replace('-','_')}"
    output_file_path = got_utils.get_output_file_path(args, output_file_name)

    np.save(output_file_path, activations)

    
    print(f'Activations for {input_file_path} saved to {output_file_path}')
