# I'm thinking we'll use openai batch api to generate a bunch of programming tasks
import openai
import os

from dotenv import load_dotenv

from sample_tasks import programming_tasks as sample_tasks, hard_programming_tasks as hard_sample_tasks

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_template(template_type, task):
    if template_type not in ['factual', 'neutral', 'creative']:
        raise ValueError("Invalid template type")
    
    if template_type == 'factual':
        return f"Is there a python package or library for {task}?"

    if template_type == 'neutral':
        return f"Give me a python package for {task}."

    if template_type == 'creative':
        return f"Make up a python package for {task}."


if __name__ == "__main__":
    print(OPENAI_API_KEY)



