import base64
import os
from io import BytesIO, BufferedReader
from unittest.mock import patch

import numpy as np

from maqcache.utils import import_replicate, import_pillow

import_replicate()
import_pillow()

import replicate
from PIL import Image

from maqcache import cache
from maqcache.adapter import replicate
from maqcache.processor.pre import get_input_str, get_input_image_file_name
from maqcache.utils.response import get_image_from_openai_url
from maqcache.manager.factory import manager_factory
from maqcache.similarity_evaluation.onnx import OnnxModelEvaluation



def test_replicate():
    test_response = {"data": [{"url": "https://raw.githubusercontent.com/zilliztech/GPTCache/dev/docs/GPTCache.png"}]}
    img_bytes = base64.b64decode(get_image_from_openai_url(test_response))
    img_file = BytesIO(img_bytes)  # convert image to file-like object
    img = Image.open(img_file)

    b_handle = BytesIO()
    img.save(b_handle, format="JPEG")
    b_handle.seek(0)
    b_handle.name = "MAQCache.png"
    expected_img_data = BufferedReader(b_handle)
    expect_answer = "maqcache"

    cache.init(pre_embedding_func=get_input_str)
    with patch("replicate.run") as mock_create:
        mock_create.return_value = expect_answer
        answer_text = replicate.run(
                        "andreasjansson/blip-2",
                        input={"image": expected_img_data,
                               "question": "Which project's architecture diagram is this?"}
                    )
        assert answer_text == expect_answer
    answer_text = replicate.run(
                    "andreasjansson/blip-2",
                    input={"image": expected_img_data,
                           "question": "Which project's architecture diagram is this?"}
                )
    assert answer_text == expect_answer

    cache.init(pre_embedding_func=get_input_image_file_name)
    with patch("replicate.run") as mock_create:
        mock_create.return_value = expect_answer
        answer_text = replicate.run(
            "andreasjansson/blip-2",
            input={"image": expected_img_data}
        )
        assert answer_text == expect_answer
    answer_text = replicate.run(
        "andreasjansson/blip-2",
        input={"image": expected_img_data}
    )
    assert answer_text == expect_answer

    faiss_file = "faiss.index"
    if os.path.isfile(faiss_file):
        os.remove(faiss_file)

    data_manager = manager_factory("sqlite,faiss", data_dir='.', vector_params={"dimension": 3})
    vector_data = np.ones((3,)).astype("float32")
    cache.init(pre_embedding_func=get_input_image_file_name,
               data_manager=data_manager,
               embedding_func=lambda data, **_: vector_data,
               similarity_evaluation=OnnxModelEvaluation())
    with patch("replicate.run") as mock_create:
        mock_create.return_value = expect_answer
        answer_text = replicate.run(
            "andreasjansson/blip-2",
            input={"image": expected_img_data,
                   "question": "Which project's architecture diagram is this?"}
        )
        assert answer_text == expect_answer
    answer_text = replicate.run(
        "andreasjansson/blip-2",
        input={"image": expected_img_data,
               "question": "Which project's architecture diagram is this?"}
    )
    assert answer_text == expect_answer
