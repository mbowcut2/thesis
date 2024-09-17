from dotenv import load_dotenv
import os
import json
from tqdm import tqdm

from pydantic import BaseModel

from openai import OpenAI

from argparse import ArgumentParser

load_dotenv()

client = OpenAI()

class LabeledQuery(BaseModel):
    query: str
    isTrue: bool

def label(queries_and_responses, model="gpt-4o-2024-08-06"):
    print("Labeling queries and responses...")
    labeled_outputs = []
    for query, response in tqdm(outputs):
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": get_prompt(response)},
            ],
            temperature=0.1,
            response_format=Claim,
        )

        labeled_outputs.append(
            {"query": query, "response": response, "label": completion.choices[0].message.parsed.dict()["isTrue"]}
        )
    print('complete!')
    return labeled_outputs




if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--input_file_path', type=str, default='data/datasets/output.json')
    argparser.add_argument('--output_file_name', type=str, default='labeled_output.json')
    argparser.add_argument('--model', type=str, default='gpt-4o-2024-08-06')
    args = argparser.parse_args()

    print(f"Loading queries and responses from {args.input_file_path}")
    with open(args.input_file_path, 'r') as f:
        queries_and_responses = json.load(f)[:3]
    
    queries_responses_labels = label(queries_and_responses, model=args.model)

    output_file_path = os.path.join('datasets', args.coding_prompt, output_file_name)
    print(f"Saving labeled outputs to {output_file_path}")
    with open(output_file_path, 'w') as f:
        json.dump(labeled_outputs, f)