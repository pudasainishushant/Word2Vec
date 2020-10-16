from gensim.models import Word2Vec
from config import model_path
import streamlit as st
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from parse_data import W2vData


training_data = W2vData()
sentences = training_data.prepare_training_text()

import pdb;pdb.set_trace()
# train model
# model = Word2Vec(sentences, size=100, window=5, min_count=2, workers=8,iter = 20)
model = Word2Vec(sentences, min_count=2, window=5,
                 sample=6e-5, alpha=0.03, min_alpha=0.007, workers=8)
model.train(sentences, total_examples=model.corpus_count,
            epochs=50, report_delay=1)

# summarize the loaded model
# summarize vocabulary
words = list(model.wv.vocab)
# access vector for one word
# save model
model.save(model_path+'/word2Vec_11000_data')

# X = model[model.wv.vocab]
# pca = PCA(n_components=10)
# result = pca.fit_transform(X)
# # create a scatter plot of the projection
# pyplot.scatter(result[:, 0], result[:, 1])
# words = list(model.wv.vocab)
# for i, word in enumerate(words):
#     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show()
print("Vocabulary size", len(words))

# model = Word2Vec.load(
#     "/home/shushant/Desktop/word2vec_training/models/newword2vec")

# most_similar = model.most_similar("python")

# for s in most_similar:
#     print("Word -- {}  similarity ---- {}".format(s[0], s[1]))

# pdb.set_trace()
