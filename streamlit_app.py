from try_word2vec import VectorScorer
from word2vec.parse_data import W2vData
from config import model_path
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk
import multiprocessing
from scipy import spatial
from matplotlib import pyplot
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import streamlit as st
from os.path import isfile, join
from os import listdir
import string
import io

# import spacy 

# spacy.cli.download("en_core_web_sm")
stop_words = set(stopwords.words('english'))

training_data = W2vData()

st.title("Document Similarity using Custom Word2Vec")
st.markdown(
    "This tool can be used to train custom Word2Vec models, visualize the model and use the trained model to view similarity between two documents similar to trained corpus.\
        One can either train word2vec model on the corpus of resume and job descriptions collected or train own word2vec model using own data/corpus. The trained model can be visualized and used to find similarity between two documents.")


# st.header("Training custom word2Vec model")

add_selectbox = st.sidebar.selectbox(
    "Select following options", ("Train custom word2vec with your own data", "Train custom word2vec using resume data pre collected", "Calculate similarity between two documents"))

if add_selectbox == "Train custom word2vec with your own data" or add_selectbox == "Train custom word2vec using resume data pre collected":
    if add_selectbox == "Train custom word2vec with your own data":
        uploaded_file = st.file_uploader("Chooose a file", type="txt")
        if uploaded_file is not None:
            data = uploaded_file.read()

    # for pdf extensions
    # data = PyPDF2.PdfFileReader(uploaded_file)
    # number_of_pages = data.numPages
    # file_content = ''
    # for i in range(0, number_of_pages):
    #     text = data.getPage(i).extractText()
    #     file_content += text

    # st.write(file_content)
    min_count = st.slider(
        label="min_count", min_value=1, max_value=6, step=1)
    st.markdown(
        "This is the minimum count of any token required to be included in the vocabulary")
    window = st.slider("Contextual window size",
                       min_value=1, max_value=6, step=1)
    st.markdown("The number of tokens on both sides of any token to be considered while predicting the target token or contecxtual size tokens. ")

    cpu_count = multiprocessing.cpu_count()
    workers = st.slider("cpu count",
                        min_value=1, max_value=cpu_count, step=1)
    st.markdown(
        "Number of workers or cpu cores which is to be used for training")
    epochs = st.slider("Epochs",
                       min_value=1, max_value=50, step=1)
    st.markdown("Number of epochs to train the model")
    approach = st.selectbox("Which approach to use for training?", [
                            "Continous Bag of Words", "Continous Skip Gram model"])

    if approach == "Continous Bag of Words":
        approach_sg = 0
    else:
        approach_sg = 1

    save_model_name = st.text_input("Save Model as : ")
    button = st.button("Train Model")
    if button:
        if add_selectbox == "Train custom word2vec with your own data":
            cleaned_data = training_data.clean_text(data)

            sent_tok_train = nltk.sent_tokenize(cleaned_data)
            sentences = [nltk.word_tokenize(sentence.translate(str.maketrans(
                '', '', string.punctuation))) for sentence in sent_tok_train]
            cleaned_sentences = []
            for sent in sentences:
                cleaned_sentence = [
                    word for word in sent if word not in stop_words]
                cleaned_sentences.append((cleaned_sentence))
        else:
            sentences = training_data.prepare_training_text()

        model = Word2Vec(sentences, min_count=min_count, window=window,
                         sample=6e-5, alpha=0.03, min_alpha=0.007, workers=workers, sg=approach_sg)
        model.train(sentences, total_examples=model.corpus_count,
                    epochs=epochs, report_delay=1)
        model.save(model_path + '/' + save_model_name)
        st.write("Custom Word2Vec model has been trained and saved under the filename : {}".format(
            save_model_name))

    onlyfiles = [f for f in listdir(model_path) if isfile(join(model_path, f))]
    model_name = st.selectbox("Model name to load", onlyfiles)

    visualize_button = st.button("Visualize model")
    if visualize_button:
        model = Word2Vec.load(
            model_path+model_name)
        X = model[model.wv.vocab]
        pca = PCA(n_components=10)
        result = pca.fit_transform(X)
        # create a scatter plot of the projection
        pyplot.scatter(result[:, 0], result[:, 1])
        words = list(model.wv.vocab)
        for i, word in enumerate(words):
            pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        st.pyplot()

    first_testing_text = st.text_input("Enter first testing string : ")
    most_similar_first_button = st.button(
        "View most similar words for first testing string in vocabulary")
    if most_similar_first_button:
        try:
            model = Word2Vec.load(model_path + model_name)
            most_similar_tokens_first = model.most_similar(
                first_testing_text.lower())
            st.write(most_similar_tokens_first)
        except KeyError as k:
            st.write("The word is not in the vocabulary")

    second_testing_text = st.text_input("Enter second testing string : ")
    most_similar_second_button = st.button(
        "View most similar words for second testing string in vocabulary")

    if most_similar_second_button:
        try:
            model = Word2Vec.load(model_path + model_name)
            most_similar_tokens_first = model.most_similar(
                second_testing_text.lower())
            st.write(most_similar_tokens_first)
        except KeyError as k:
            st.write("The testing string is not in the vocabulary")

    similarity_button = st.button("View similarity of above two tokens")
    if similarity_button:
        vs = VectorScorer(
            model_path + model_name)

        test_string1_tokenized = word_tokenize(first_testing_text.lower())
        test_string1_without_stopwords = [
            word for word in test_string1_tokenized if not word in stopwords.words()]
        test_string2_tokenized = word_tokenize(second_testing_text.lower())
        test_string2_without_stopwords = [
            word for word in test_string2_tokenized if not word in stopwords.words()]
        first_string_vector = vs.create_vector(test_string1_without_stopwords)
        second_string_vector = vs.create_vector(test_string2_without_stopwords)
        similarity = vs.calculate_similarity(
            first_string_vector, second_string_vector)
        st.write("Similarity calculated is ", similarity)
        st.write("Word Embedding calculated for first testing string",
                 first_string_vector)
        st.write("Word Embedding calculated for second testing string",
                 second_string_vector)

else:
    uploaded_file1 = st.file_uploader(
        "Upload first testing document", type="txt")
    if uploaded_file1 is not None:
        data1 = uploaded_file1.read()
        cleaned_data1 = training_data.clean_text(data1)

        data_1_tokenized = word_tokenize(cleaned_data1)
        data_1_final = [
            word for word in data_1_tokenized if not word in stopwords.words()]
        # st.write(data_1_final)

    uploaded_file2 = st.file_uploader(
        "Upload second testing document", type="txt")
    if uploaded_file2 is not None:
        data2 = uploaded_file2.read()
        cleaned_data2 = training_data.clean_text(data2)

        data_2_tokenized = word_tokenize(cleaned_data2)
        data_2_final = [
            word for word in data_2_tokenized if not word in stopwords.words()]

        # st.write(data_2_final)
    onlyfiles = [f for f in listdir(model_path) if isfile(join(model_path, f))]
    model_name = st.selectbox("Select among the trained models", onlyfiles)

    similarity_button = st.button("View similarity of the two documents")
    if similarity_button:
        vs = VectorScorer(
            model_path + model_name)

        first_string_vector = vs.create_vector(data_1_final)
        second_string_vector = vs.create_vector(data_2_final)
        similarity = vs.calculate_similarity(
            first_string_vector, second_string_vector)
        st.write("Similarity calculated is ", similarity)
