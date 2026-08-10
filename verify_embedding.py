import sys
sys.path.insert(0, 'src')
import embeddings.embedding_model as embedding_module

class BrokenEmbeddings:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('download failed')

embedding_module.HuggingFaceEmbeddings = BrokenEmbeddings
model = embedding_module.get_embedding_model()
assert model is not None
assert len(model.embed_query('hello')) == 128
print('fallback ok')
