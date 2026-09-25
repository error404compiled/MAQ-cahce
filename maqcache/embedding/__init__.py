__all__ = [
    "OpenAI",
    "Huggingface",
    "SBERT",
    "Cohere",
    "Onnx",
    "FastText",
    "Data2VecAudio",
    "Timm",
    "ViT",
    "LangChain",
    "Rwkv",
    "PaddleNLP",
    "UForm",
]


from maqcache.utils.lazy_import import LazyImport

openai = LazyImport("openai", globals(), "maqcache.embedding.openai")
huggingface = LazyImport("huggingface", globals(), "maqcache.embedding.huggingface")
sbert = LazyImport("sbert", globals(), "maqcache.embedding.sbert")
onnx = LazyImport("onnx", globals(), "maqcache.embedding.onnx")
cohere = LazyImport("cohere", globals(), "maqcache.embedding.cohere")
fasttext = LazyImport("fasttext", globals(), "maqcache.embedding.fasttext")
data2vec = LazyImport("data2vec", globals(), "maqcache.embedding.data2vec")
timm = LazyImport("timm", globals(), "maqcache.embedding.timm")
vit = LazyImport("vit", globals(), "maqcache.embedding.vit")
langchain = LazyImport("langchain", globals(), "maqcache.embedding.langchain")
rwkv = LazyImport("rwkv", globals(), "maqcache.embedding.rwkv")
paddlenlp = LazyImport("paddlenlp", globals(), "maqcache.embedding.paddlenlp")
uform = LazyImport("uform", globals(), "maqcache.embedding.uform")


def Cohere(model="large", api_key=None):
    return cohere.Cohere(model, api_key)


def OpenAI(model="text-embedding-ada-002", api_key=None):
    return openai.OpenAI(model, api_key)


def Huggingface(model="distilbert-base-uncased"):
    return huggingface.Huggingface(model)


def SBERT(model="all-MiniLM-L6-v2"):
    return sbert.SBERT(model)


def Onnx(model="MAQCache/paraphrase-albert-onnx"):
    return onnx.Onnx(model)


def FastText(model="en", dim=None):
    return fasttext.FastText(model, dim)


def Data2VecAudio(model="facebook/data2vec-audio-base-960h"):
    return data2vec.Data2VecAudio(model)


def Timm(model="resnet50", device="default"):
    return timm.Timm(model, device)


def ViT(model="google/vit-base-patch16-384"):
    return vit.ViT(model)


def LangChain(embeddings, dimension=0):
    return langchain.LangChain(embeddings, dimension)


def Rwkv(model="sgugger/rwkv-430M-pile"):
    return rwkv.Rwkv(model)


def PaddleNLP(model="ernie-3.0-medium-zh"):
    return paddlenlp.PaddleNLP(model)


def UForm(model="unum-cloud/uform-vl-multilingual", embedding_type="text"):
    return uform.UForm(model, embedding_type)
