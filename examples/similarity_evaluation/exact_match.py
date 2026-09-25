from maqcache.adapter import openai
from maqcache import cache
from maqcache.similarity_evaluation.exact_match import ExactMatchEvaluation


def run():
    cache.init(similarity_evaluation=ExactMatchEvaluation())
    cache.set_openai_key()

    answer = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user', 'content': 'what is chatgpt'}
        ],
    )
    print(answer)


if __name__ == '__main__':
    run()
