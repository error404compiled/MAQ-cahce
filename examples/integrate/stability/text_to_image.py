import io
import os
import time

from PIL import Image

from maqcache import cache
from maqcache.adapter.stability_sdk import StabilityInference, generation
from maqcache.embedding import Onnx
from maqcache.manager.factory import manager_factory
from maqcache.processor.pre import get_prompt
from maqcache.similarity_evaluation.distance import SearchDistanceEvaluation

# init maqcache
onnx = Onnx()
data_manager = manager_factory('sqlite,faiss,local',
                               data_dir='/',
                               vector_params={'dimension': onnx.dimension},
                               object_params={'path': './images'}
                               )
cache.init(
    pre_embedding_func=get_prompt,
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation()
    )

# run with maqcache
api_key = os.getenv('STABILITY_KEY', 'key-goes-here')

stability_api = StabilityInference(
    key=os.environ['STABILITY_KEY'], # API Key reference.
    verbose=False, # Print debug messages.
    engine='stable-diffusion-xl-beta-v2-2-2', # Set the engine to use for generation.
)

start = time.time()
answers = stability_api.generate(
    prompt='a cat sitting besides a dog',
    width=256,
    height=256
    )

for resp in answers:
    for artifact in resp.artifacts:
        if artifact.type == generation.ARTIFACT_IMAGE:
            img = Image.open(io.BytesIO(artifact.binary))
            assert img.size == (256, 256)
print('Time elapsed 1:', time.time() - start)

start = time.time()
answers = stability_api.generate(
    prompt='a dog and a dog sitting together',
    width=512,
    height=512
    )

for resp in answers:
    for artifact in resp.artifacts:
        if artifact.type == generation.ARTIFACT_IMAGE:
            img = Image.open(io.BytesIO(artifact.binary))
            assert img.size == (512, 512)
print('Time elapsed 2:', time.time() - start)