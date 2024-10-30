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

class BooleanResponse(BaseModel):
    isTrue: bool

def get_chat_prompt(response):
    return f"The following is a conversation with an AI assistant. If the assistant mentions a tool, package or library that does not exist or that cannot be used for the programming task, please mark as incorrect. Otherwise, mark as correct. If the AI assistant responds that it doesn't know the answer, mark as correct.\n\n {response}"
    # return f"The following is a conversation with an AI assistant. Is the information provided by the AI correct? If the AI indicates that it doesn't know, then please mark as correct. Only mark as incorrect if the AI responds with a package or library that does not exist.\n\n {response}"

def get_instruct_prompt(response):
    return f"The following is a conversation with an AI assistant. If the assistant mentions a tool, package or library that does not exist or that cannot be used for the programming task, please mark as incorrect. Otherwise, mark as correct. If the AI assistant responds that it doesn't know the answer, mark as correct.\n\n {response}"


def get_base_prompt(response):
    lines = response.split('\n')
    question = lines[0]
    answer = lines[1]
    if len(lines) > 2:
        for line in lines[2:]:
            if 'Q:' in line:
                break
            answer += line

    q_a_pair = ''.join([question, '\n', answer])
    return f"The following is a question and answer response from an AI assistant. If the assistant mentions a tool, package or library that does not exist, please mark as incorrect. Otherwise, mark as correct.\n\n {q_a_pair}"
    # return f"The following is a question and answer response from an AI assistant. Is the information provided by the AI correct? If the AI indicates that it doesn't know, then please mark as correct. Only mark as incorrect if the AI responds with a package or library that does not exist.\n\n {q_a_pair}"

def label(data, args, openai_model="gpt-4o-2024-08-06", model='', double_check=False):
    print("Labeling queries and responses...")
    labeled_data = []
    num_requests = 0
    for d in tqdm(data):
        if not d['label']:
            num_requests += 1      

            completion = client.beta.chat.completions.parse(
                model=openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Do any of the tools discussed here not exist? {d['response']}"},
                ],
                temperature=0.1,
                response_format=BooleanResponse,
            )
            # reverse the boolean because we are asking if the tools do not exist
            response_dict = {"query": d["query"], "response": d["response"], "label": not completion.choices[0].message.parsed.dict()["isTrue"], "explanation": d["explanation"]}
            # response_dict['label'] = not completion.choices[0].message.parsed.dict()["isTrue"]
            labeled_data.append(response_dict)
        else:
            labeled_data.append(d)

    print('complete!')
    print(f'Issued {num_requests} requests')
    return labeled_data




if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--openai_model', type=str, default='gpt-4o-2024-08-06')
    argparser.add_argument('--coding_prompt', type=str, default='factual')
    argparser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
    argparser.add_argument('--dataset', type=str, default='easy_100')
    args = argparser.parse_args()

    input_file_path = get_output_file_path(args, 'labeled') + '.json' # this is bad naming

    print(f"Loading queries and responses from {input_file_path}")
    
    with open(input_file_path, 'r') as f:
        data = json.load(f)
    
    labeled_data = label(data, args, openai_model=args.openai_model, model=args.model)

    output_file_path = get_output_file_path(args, 'labeled_double_checked') + '.json'
    print(f"Saving labeled outputs to {output_file_path}")
    with open(output_file_path, 'w') as f:
        json.dump(labeled_data, f)