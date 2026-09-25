import os
import time

import openai


def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']


# Before running this case, make sure the OPENAI_API_KEY environment variable is set

question = 'what‘s chatgpt'

# OpenAI API original usage
openai.api_key = os.getenv('OPENAI_API_KEY')
start_time = time.time()
response = openai.ChatCompletion.create(
  model='gpt-3.5-turbo',
  messages=[
    {
        'role': 'user',
        'content': question
    }
  ],
)
print(f'Question: {question}')
print('Time consuming: {:.2f}s'.format(time.time() - start_time))
print(f'Answer: {response_text(response)}\n')

# MAQCache exact matching usage
print('MAQCache exact matching example.....')
print('Cache loading.....')

# To use MAQCache, that's all you need
# -------------------------------------------------
from maqcache import cache
from maqcache.adapter import openai

cache.init()
cache.set_openai_key()
# -------------------------------------------------

question = 'what is github'
for _ in range(2):
    start_time = time.time()
    response = openai.ChatCompletion.create(
      model='gpt-3.5-turbo',
      messages=[
        {
            'role': 'user',
            'content': question
        }
      ],
    )
    print(f'Question: {question}')
    print('Time consuming: {:.2f}s'.format(time.time() - start_time))
    print(f'Answer: {response_text(response)}\n')

# MAQCache similar search usage
print('MAQCache similar search example.....')
print('Cache loading.....')

from maqcache import cache
from maqcache.adapter import openai
from maqcache.embedding import Onnx
from maqcache.manager import get_data_manager, VectorBase
from maqcache.similarity_evaluation.distance import SearchDistanceEvaluation

onnx = Onnx()
vector_base = VectorBase('faiss', dimension=onnx.dimension)
data_manager = get_data_manager('sqlite', vector_base)
cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    )
cache.set_openai_key()

questions = [
    'what is github',
    'can you explain what GitHub is',
    'can you tell me more about GitHub'
    'what is the purpose of GitHub'
]

for question in questions:
    for _ in range(2):
        start_time = time.time()
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {
                    'role': 'user',
                    'content': question
                }
            ],
        )
        print(f'Question: {question}')
        print('Time consuming: {:.2f}s'.format(time.time() - start_time))
        print(f'Answer: {response_text(response)}\n')
