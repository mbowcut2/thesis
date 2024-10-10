from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn

import argparse

import json
from tqdm import tqdm
import os

from utils import get_output_file_path, get_input_file_path

LLAMA_PATH = '../../models/llama2/'
DEVICE = 'cuda:5'
MODELS_PATH = '../../models/'

def get_chat_template(system_prompt, user_message):


    return f'''<s>[INST] <<SYS>>
{ system_prompt }
<</SYS>>

{ user_message } [/INST]
'''

def get_instruct_template(system_prompt, user_message):
    return f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

def get_QA_template(system_prompt, user_message):
    return f'''Q: {user_message}
A:'''

def get_model_output(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

    output = model.generate(input_ids, max_length=200, pad_token_id=tokenizer.eos_token_id)

    return tokenizer.decode(output[0], skip_special_tokens=False) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input', type=str, default='The meaning of life is')
    parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
    parser.add_argument('--system_prompt', type=str, default='You are a helpful AI assistant.')
    parser.add_argument('--coding_prompt', type=str, default='factual')
    parser.add_argument('--interactive', action='store_true')
    # parser.add_argument('--input_file_name', type=str, default='')
    parser.add_argument('--dataset', type=str, default='easy_100')
    # parser.add_argument('--output_file_path', type=str, default='output.json')
    args = parser.parse_args()

    model_slug = args.model.replace('-', '_')

    output_file_name = f"responses_{args.model.replace('-', '_')}"

    # model = AutoModelForCausalLM.from_pretrained(
    #     MODELS_PATH + args.model,
    #     device_map= 'auto'
    # )
    # tokenizer = AutoTokenizer.from_pretrained(MODELS_PATH + args.model)

    model = LlamaForCausalLM.from_pretrained(
        MODELS_PATH + args.model,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(MODELS_PATH + args.model)

    if args.coding_prompt == 'factual':
        coding_prompt = 'What is the name of a python package or library that can be used to '
    elif args.coding_prompt == 'neutral':
        coding_prompt = 'Give me the name of a python package that can be used to '
    elif args.coding_prompt == 'creative':
        coding_prompt = 'Make up the name for a python package that could be used to '
    elif args.coding_prompt == 'yes_no':
        coding_prompt = 'Is there a python package that can be used to '
    else:
        coding_prompt = args.coding_prompt

    if args.interactive:
        print('Type "exit" to quit')
        while True:
            # my_input = input(f'{coding_prompt} \nEnter your input: ')
            my_input = input(f'Enter your input: ')
            if my_input == 'exit':
                break
            # prompt = coding_prompt + my_input
            prompt = my_input
            # chat_input = get_chat_template(args.system_prompt, prompt)
            print(get_model_output(prompt, model, tokenizer))

    else:
        if 'chat' in args.model:
            get_template = get_chat_template
        elif 'Instruct' in args.model:
            get_template = get_instruct_template
        else:
            get_template = get_QA_template
        if args.dataset:
            input_file_path = get_input_file_path(args) + '.json'
            print(f'Running inference on {args.model} with {input_file_path} and coding prompt: {coding_prompt}')
            with open(input_file_path, 'r') as f:
                data = json.load(f)
                try:
                    tasks = data.get('tasks')
                except:
                    tasks = data
                outputs = []
                for task in tqdm(tasks):
                    prompt = coding_prompt + task.lower()
                    chat_input = get_template(args.system_prompt, prompt)
                    outputs.append([prompt, get_model_output(chat_input, model, tokenizer)])


            output_file_path = get_output_file_path(args, 'data') + '.json'
            with open(output_file_path, 'w') as f:
                json.dump(outputs, f)
        else:
            print('Please specify an input file or run in interactive mode')
