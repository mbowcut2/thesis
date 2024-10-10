# I'm thinking we'll use openai batch api to generate a bunch of programming tasks
from openai import OpenAI

import os
import json
from pydantic import BaseModel
from dotenv import load_dotenv
from argparse import ArgumentParser

import time


load_dotenv()

client = OpenAI()


class TaskList(BaseModel):
    tasks: list[str]
    length: int

def generate_task_list(model, prompt):
    print(f"Using {model} to generate a list of programming tasks...")
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        response_format=TaskList,
    )
    print('completed')
    return completion.choices[0].message.parsed.dict()




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--openai_model', type=str, default='gpt-4o-2024-08-06')
    parser.add_argument('--n_tasks', type=int, default=5)
    parser.add_argument('--output_file', type=str, default='')
    parser.add_argument('--random_seed', type=int, default=int(time.time()))
    parser.add_argument('--prompt', type=str, default='')
    args = parser.parse_args()

    if args.prompt == '':
        prompt = f"Using random seed {args.random_seed}, give me a list of {args.n_tasks} general programming tasks like: handle HTTP requests, encrypt and decrypt data, calculate the derivative of a function, etc."
    else:
        prompt = args.prompt

    task_list = generate_task_list(args.openai_model, prompt)
    print(f'Prompt: {prompt}')
    if int(task_list['length']) != len(task_list['tasks']):
        print(f"Warning: expected {task_list['length']} tasks, but got {len(task_list['tasks'])}")
    
    if args.output_file == '':
        output_path = os.path.join('datasets', f"tasks_{args.random_seed}.json")
    else:
        output_path = os.path.join('datasets', args.output_file)
    with open(output_path, 'w') as f:
        json.dump(task_list, f)
    print(f"tasks saved to {output_path}")




