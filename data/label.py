from dotenv import load_dotenv
import os
import json
from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI
from argparse import ArgumentParser

from utils import get_output_file_path

load_dotenv()

client = OpenAI()

class LabeledQuery(BaseModel):
    query: str
    isTrue: bool

def get_prompt(response):
    return f'The following is a conversation with an AI assistant. Is the information provided by the AI correct?\n\n {response}'

def label(queries_and_responses, model="gpt-4o-2024-08-06"):
    print("Labeling queries and responses...")
    labeled_data = []
    for query, response in tqdm(queries_and_responses):
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": get_prompt(response)},
            ],
            temperature=0.1,
            response_format=LabeledQuery,
        )

        labeled_data.append(
            {"query": query, "response": response, "label": completion.choices[0].message.parsed.dict()["isTrue"]}
        )
    print('complete!')
    return labeled_data




if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--model', type=str, default='gpt-4o-2024-08-06')
    argparser.add_argument('--coding_prompt', type=str, default='factual')
    argparser.add_argument('--dataset', type=str, default='easy_100')
    args = argparser.parse_args()

    input_file_path = os.path.join('datasets', args.dataset)

    print(f"Loading queries and responses from {args.input_file_path}")
    with open(args.input_file_path, 'r') as f:
        queries_and_responses = json.load(f)
    
    labeled_data = label(queries_and_responses, model=args.model)

    output_file_path = get_output_file_path(args, 'labeled.json')
    print(f"Saving labeled outputs to {output_file_path}")
    with open(output_file_path, 'w') as f:
        json.dump(labeled_data, f)