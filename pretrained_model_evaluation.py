from gensim.models import KeyedVectors
# Load vectors directly from the file
model = KeyedVectors.load_word2vec_format(
    '/home/shushant/Desktop/word2vec_training/models/GoogleNews-vectors-negative300 (1).bin.gz', binary=True)
# Access vectors for specific words with a keyed lookup:
vector = model['easy']
# see the shape of the vector (300,)
vector.shape
