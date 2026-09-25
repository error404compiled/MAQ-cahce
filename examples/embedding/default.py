from maqcache.adapter import openai
from maqcache import cache
from maqcache.embedding.string import to_embeddings as string_embedding


def run():
    cache.init(embedding_func=string_embedding)
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
