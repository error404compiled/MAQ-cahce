import numpy as np

from maqcache import cache
from maqcache.adapter import openai
from maqcache.manager import CacheBase, VectorBase, get_data_manager
from maqcache.similarity_evaluation.distance import SearchDistanceEvaluation

d = 8


def mock_embeddings(data, **kwargs):
    return np.random.random((d, )).astype('float32')


def run():
    vector_stores = [
        'faiss',
        'milvus',
        'chromadb',
        'docarray',
        'redis',
        'weaviate',
    ]
    for vector_store in vector_stores:
        cache_base = CacheBase('sqlite')
        vector_base = VectorBase(vector_store, dimension=d)
        data_manager = get_data_manager(cache_base, vector_base)

        cache.init(
            embedding_func=mock_embeddings,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(),
        )
        cache.set_openai_key()

        answer = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[{'role': 'user', 'content': 'what is chatgpt'}],
        )
        print(answer)


if __name__ == '__main__':
    run()
