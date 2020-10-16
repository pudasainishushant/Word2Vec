from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy import spatial
import numpy as np
import nltk
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))


class VectorScorer():
    '''
    scores the resume using the
    word2vec gensim model
    '''

    def __init__(self, model_path):
        '''
        This constructor loads the word2vec model that
        we will be using for vectorizing of embedding the
        contents of the resume and job description
        '''
        self.word2vec = Word2Vec.load(
            model_path)

    def create_vector(self, token_list):
        '''
        Takes the list of the tokens and returns their embedded vectors
        :param resume_token_list:type list of list
        :return:doc_vectors :type list of list
        '''
        doc_word2vec = list()

        tokenized = [word_tokenize(token) for token in token_list]
        flat_list = [item for sublist in tokenized for item in sublist]
        for token in flat_list:
            try:
                doc_word2vec.append(self.word2vec[token])
            except KeyError as k:
                print(k)
        doc_vectors = (np.mean(doc_word2vec, axis=0))
        return doc_vectors

    def calculate_similarity(self, first_vector, second_vector):
        '''
        this function makes the use of cosine distance to measure the similarity
        between job and cv
        :param job_description_vector:type:list
        :param cv_vectors_list :type: list of list
        :return: score_list
        '''
        score = ((1 - spatial.distance.cosine(first_vector, second_vector)))
        if score < 0:
            score = 0
        elif score > 100:
            score = 95
        return score


# if __name__ == "__main__":
#     vs = VectorScorer()
#     while True:
#         test_string1 = str(input("Enter first test string : ")).lower()
#         test_string1_tokenized = word_tokenize(test_string1)
#         test_string1_without_stopwords = [
#             word for word in test_string1_tokenized if not word in stopwords.words()]
#         test_string2 = str(input("Enter second test string : ")).lower()
#         test_string2_tokenized = word_tokenize(test_string2)
#         test_string2_without_stopwords = [
#             word for word in test_string2_tokenized if not word in stopwords.words()]
#         first_string_vector = vs.create_vector(test_string1_without_stopwords)
#         second_string_vector = vs.create_vector(test_string2_without_stopwords)
#         similarity = vs.calculate_similarity(
#             first_string_vector, second_string_vector)
#         print(similarity)
