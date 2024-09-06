# I'm thinking we'll use openai batch api to generate a bunch of programming tasks
import openai
import os

from dotenv import load_dotenv

from sample_tasks import programming_tasks as sample_tasks, hard_programming_tasks as hard_sample_tasks

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


if __name__ == "__main__":
    print(OPENAI_API_KEY)



