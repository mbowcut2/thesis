from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn

import argparse

LLAMA_PATH = '../../llama2/'
DEVICE = 'cuda:5'

def get_chat_template(system_prompt, user_message):


    return f'''<s>[INST] <<SYS>>
{ system_prompt }
<</SYS>>

{ user_message } [/INST]
'''


def get_model_output(input, model, tokenizer):
    input_ids = tokenizer.encode(input, return_tensors='pt').to(model.device)

    output = model.generate(input_ids, max_length=200)

    # breakpoint()

    print(tokenizer.decode(output[0], skip_special_tokens=True))
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input', type=str, default='The meaning of life is')
    parser.add_argument('--model', type=str, default='Llama-2-7b-hf')
    parser.add_argument('--system_prompt', type=str, default='You are a helpful AI assistant.')
    parser.add_argument('--coding_prompt', type=str, default='')
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH + args.model,
        device_map= 'auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH + args.model)

    while True:
        if args.coding_prompt == 'factual':
            coding_prompt = 'What is the name of a python package or library that can be used to '
        elif args.coding_prompt == 'neutral':
            coding_prompt = 'Give me the name of a python package that can be used to '
        elif args.coding_prompt == 'creative':
            coding_prompt = 'Make up the name for a python package that could be used to '
        else:
            coding_prompt = ''
        my_input = input('Enter your input: ')
        if my_input == 'exit':
            break
        prompt = coding_prompt + my_input
        chat_input = get_chat_template(args.system_prompt, prompt)
        get_model_output(chat_input, model, tokenizer)


    # input = args.input

    # input_ids = tokenizer.encode(input, return_tensors='pt').to(DEVICE)

    # output = model.generate(input_ids, max_length=50, num_return_sequences=3, temperature=0.7)

    # for i in range(3):
    #     print(tokenizer.decode(output[i], skip_special_tokens=True))