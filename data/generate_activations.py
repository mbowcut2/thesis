from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn

import argparse

import json
from tqdm import tqdm
import os

import numpy as np

from utils import get_output_file_path, get_input_file_path

LLAMA_PATH = '../../models/llama2/'
DEVICE = 'cuda:5'
MODELS_PATH = '../../models/'
    

def get_model_activations(input, model, tokenizer):
    inputs = tokenizer(input, return_tensors='pt')
    outputs = model(**inputs, output_hidden_states=True)
    activations = [x[:, -1, :] for x in outputs.hidden_states]
    return torch.cat(activations).detach().cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input', type=str, default='The meaning of life is')
    parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
    parser.add_argument('--coding_prompt', type=str, default='factual')
    parser.add_argument('--dataset', type=str, default='easy_100')
    args = parser.parse_args()

    input_file_path = get_output_file_path(args, "labeled") + ".json"

    # model = AutoModelForCausalLM.from_pretrained(
    #     LLAMA_PATH + args.model,
    #     device_map= 'auto'
    # )
    # tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH + args.model)

    model = LlamaForCausalLM.from_pretrained(
        MODELS_PATH + args.model,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(MODELS_PATH + args.model)

    with open(input_file_path, 'r') as f:
        data = json.load(f)

    activations = np.stack([get_model_activations(d['query'], model, tokenizer) for d in data])

    output_file_path = get_output_file_path(args, 'acts') + '.npy'

    np.save(output_file_path, activations)

    
    print(f'Activations for {input_file_path} saved to {output_file_path}')
