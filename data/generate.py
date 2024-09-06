# I'm thinking we'll use openai batch api to generate a bunch of programming tasks
import openai
import os

from sample_tasks import programming_tasks as sample_tasks, hard_programming_tasks as hard_sample_tasks

openai.api_key = None # TODO: set your api key here



