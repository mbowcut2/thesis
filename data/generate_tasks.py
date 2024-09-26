# I'm thinking we'll use openai batch api to generate a bunch of programming tasks
from openai import OpenAI

import os
import json
from pydantic import BaseModel
from dotenv import load_dotenv
from argparse import ArgumentParser


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
    parser.add_argument('--prompt', type=str, default='"Give me a list of 10 programming tasks.')
    parser.add_argument('--output_file', type=str, default='tasks.json')
    args = parser.parse_args()

    task_list = generate_task_list(args.openai_model, args.prompt)
    print(f'Prompt: {args.prompt}')
    if int(task_list['length']) != len(task_list['tasks']):
        print(f"Warning: expected {task_list['length']} tasks, but got {len(task_list['tasks'])}")

    output_path = os.path.join('datasets', args.output_file)
    with open(output_path, 'w') as f:
        json.dump(task_list, f)
    print(f"tasks saved to {output_path}")




