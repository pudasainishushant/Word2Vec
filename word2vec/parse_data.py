import nltk
from nltk.corpus import stopwords
import config as cfg
import fitz
import docx2txt
import pandas as pd
import string
import os
import re
import spacy
print('spaCy Version: %s' % (spacy.__version__))
spacy_nlp = spacy.load('en_core_web_sm')

class W2vData:

    def __init__(self):
        '''
        initializer for initializing the
        data path and stopwords for further use
        '''
        self.input_data = cfg.data_path  # for path, subdirs, files in os.walk(
        self.spacy_stopwords = set(spacy.lang.en.stop_words.STOP_WORDS)
        self.stopwords = set(stopwords.words('english')).union(self.spacy_stopwords)

    def clean_text(self, text):
        '''
        accepts the plain text and makes
        use of regex for cleaning the noise
        :param: text :type:str
        :return:cleaned text :type str
        '''
        try:
            text = text.lower()
            text = ''.join([i for i in text if not i.isdigit()])
            text = re.sub(
                r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?', '', text)
            text = re.sub(r'[|:}{=]', ' ', text)
            text = re.sub(r'[;]', ' ', text)
            text = re.sub(r'[\n]', ' ', text)
            text = re.sub(r'[\t]', ' ', text)
            text = re.sub(r'[[[]', ' ', text)
            text = re.sub(r'[]]]', ' ', text)
            text = re.sub(r'[-]', ' ', text)
            text = re.sub(r'[+]', ' ', text)
            text = re.sub(r'[*]', ' ', text)
            text = re.sub(r'[/]', ' ', text)
            text = re.sub(r'[//]', ' ', text)
            text = re.sub(r'[@]', ' ', text)
            text = re.sub(r'[,]', ' ', text)
            text = re.sub(r'[)]', ' ', text)
            text = re.sub(' +', ' ', text)
            text = re.sub('\n+', '\n', text)
            text = re.sub('\t+', '\t', text)
            text = [i.strip() for i in text.splitlines()]
            text = '\n'.join(text)
            text = re.sub('\n+', '\n', text)
            text = re.sub(r'[-]', ' ', text)
            text = re.sub(r'[(]', ' ', text)
            text = re.sub(' + ', ' ', text)
            text = text.encode('ascii', errors='ignore').decode("utf-8")
            return text
        except Exception as e:
            print("Error while cleaning text --->{}".format(e))
            pass

    def pdf_to_text(self, file_path):
        '''
        function that takes datapath
        extracts the plain text from pdf
        for training the word to vec model
        :param file_path :type str
        :return:text   :type str
        '''
        doc = fitz.open(file_path)
        number_of_pages = doc.pageCount
        text = ''
        for i in range(0, number_of_pages):
            page = doc.loadPage(i)
            pagetext = page.getText("text")
            text += pagetext
        text = clean_text(text, dolower)
        return text

    def docx_to_text(self, file_path):
        '''
        function for extracting plain text
        from the docx files
        :param file_path :type str
        :return:text     :type str
        '''
        text = ""
        text += docx2txt.process(file_path)
        text = self.clean_text(text)
        return text

    def txt_to_text(self, file_path):
        '''
        function for extracting plain text from
        txt files
        :param file_path :type str
        :return:text     :type str
        '''
        text = ""
        with open(file_path, mode='r', encoding='unicode_escape', errors='strict', buffering=1) as file:
            data = file.read()
        text += data
        text = self.clean_text(text)
        return text

    def csv_to_text(self, file_path):
        '''
        It takes the csv which contains the list of the
        jobs description and takes the description and put
        use them to prepare data.
        :param file_path :type str
        :return: list of tokenized sentences :type list
        '''
        dataframe = pd.read_csv(file_path)
        list_of_descriptions = dataframe['Description'].values()
        return list_of_descriptions

    def prepare_training_text(self):
        '''
        converts the plain text into the form
        that is compatible for training the
        word2vec model
        :return: list of cleaned tokenized sentences :type list
        '''

        reader_choice = {'.pdf': self.pdf_to_text,
                         '.docx': self.docx_to_text,
                         '.txt': self.txt_to_text,
                         '.csv': self.csv_to_text
                         }

        training_set = []
        files = []

        for r, d, f in os.walk(cfg.data_path):
            files = [os.path.join(r, file) for file in f]
        import pdb;pdb.set_trace()
        exts = set()
        for doc in files:
            name, ext = os.path.splitext(doc)
            exts.add(ext)
            # if ext == ".csv":
            #     print(doc)
            try:
                content = reader_choice.get(ext)(doc)

                if content and ext != '.csv':
                    training_set.append(content)
                else:
                    print("i am in the csv field")
                    training_set.extend(content)
            except Exception as e:
                print(e)

        training_data = ' '.join(training_set)

        sent_tok_train = nltk.sent_tokenize(training_data)
        sentences = [nltk.word_tokenize(sentence.translate(str.maketrans(
            '', '', string.punctuation))) for sentence in sent_tok_train]
        cleaned_sentences = []
        for sent in sentences:
            cleaned_sentence = [
                word for word in sent if word not in self.stopwords]
            cleaned_sentences.append((cleaned_sentence))
        return cleaned_sentences
