from io import BytesIO

import requests

from maqcache.adapter.api import _get_model
from maqcache.utils import import_pillow, import_vit


def test_timm():
    import_vit()
    import_pillow()

    from PIL import Image
    from maqcache.embedding import ViT

    url = 'https://raw.githubusercontent.com/zilliztech/GPTCache/main/docs/GPTCache.png'
    image_bytes = requests.get(url).content
    image_data = BytesIO(image_bytes)  # Convert image to file-like object
    image = Image.open(image_data)
    encoder = ViT(model="google/vit-base-patch16-384")
    embed = encoder.to_embeddings(image)
    assert len(embed) == encoder.dimension

    encoder = _get_model(model_src="vit")
    embed = encoder.to_embeddings(image)
    assert len(embed) == encoder.dimension

if __name__ == "__main__":
    test_timm()