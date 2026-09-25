from maqcache.adapter import openai
from maqcache import cache
from maqcache.manager import get_data_manager, VectorBase
from maqcache.similarity_evaluation import SequenceMatchEvaluation
from maqcache.processor.pre import concat_all_queries
from maqcache.embedding import Onnx
from maqcache import Config


def run():
    onnx = Onnx()

    vector_base = VectorBase('faiss', dimension=onnx.dimension)
    data_manager = get_data_manager('sqlite', vector_base)

    cache.init(embedding_func=onnx.to_embeddings,
               pre_embedding_func=concat_all_queries,
               data_manager=data_manager,
               similarity_evaluation=SequenceMatchEvaluation([0.1, 0.2, 0.7], 'onnx'),
               config=Config(context_len=3, skip_list=['system', 'assistant'])
               )
    cache.set_openai_key()

    answer = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': 'you are a helpful chatbot.'},
            {'role': 'user', 'content': 'query1'},
            {'role': 'assistant', 'content': 'answer1'},
            {'role': 'user', 'content': 'query2'},
            {'role': 'assistant', 'content': 'answer2'},
            {'role': 'user', 'content': 'query3'},
            {'role': 'assistant', 'content': 'answer3'}
        ]
    )
    print(answer)


if __name__ == '__main__':
    run()
