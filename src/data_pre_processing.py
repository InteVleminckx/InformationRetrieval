import ast
import pickle

import nltk
import pandas as pd
from joblib import Memory
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

memory = Memory("cached", verbose=0)


@memory.cache
def safe_eval(text):
    """
    This function evaluates an expression node or a string consisting of a Python literal or container display.
    https://www.educative.io/answers/what-is-astliteralevalnodeorstring-in-python
    :param text: input text to evaluate
    :return:
    """
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


@memory.cache
def get_preprocessed_data(path):
    # downloading punkt and stopwords
    nltk.download('punkt')
    nltk.download('stopwords')

    # retrieve stop words
    stop_words = set(stopwords.words("english"))

    # create stemmer
    stemmer = SnowballStemmer("english")

    # Converting the dataset to a dataframe
    df = pd.read_csv(path)

    # Retrieving the titles
    titles = df['Title'].to_list()

    # Convert the text to lists
    df["Sections"] = df["Sections"].apply(safe_eval)

    data = {}
    tokenized_docs_stringed = []
    tokenized_docs_listed = []

    # Looping over the rows of data
    for index, row in df.iterrows():
        title = row["Title"]
        sections = row["Sections"]

        # Merging the content of the section as one string and removing escape characters + convert to lowercase
        merged_sections = ''.join(
            paragraph.replace(r"\\u", r"\u").replace("\n", "").encode('utf-8', 'ignore').decode('utf-8').lower()
            for section in sections for paragraph in section
        )

        tokenized_document = {
            "string": "",
            "list": [],
        }

        # convert to tokens, remove stop words + stem each word
        for word in word_tokenize(merged_sections):
            if word not in stop_words:
                word = stemmer.stem(word)
                # creating string of the words or a list
                tokenized_document["string"] += f" {word}"
                tokenized_document["list"].append(word)

        tokenized_docs_stringed.append(tokenized_document["string"])
        tokenized_docs_listed.append(tokenized_document["list"])
        data[title] = {
            "text": merged_sections,
            "tok-stringed": tokenized_document["string"],
            "tok-listed": tokenized_document["list"]
        }

    return titles, data, tokenized_docs_stringed, tokenized_docs_listed


def get_ground_truth(path):
    with open(path, "rb") as f:
        binary_data = f.read()
        return dict(pickle.loads(binary_data))


class DataPreProcessing:

    def __init__(self, titles, data, tokenized_docs_stringed, tokenized_docs_listed):
        self.titles = titles
        self.data = data
        self.tokenized_documents_stringed = tokenized_docs_stringed
        self.tokenized_documents_listed = tokenized_docs_listed
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = SnowballStemmer("english")

    @staticmethod
    def preprocess_text(text):
        """
        This function preprocesses the text by removing escape characters and converting the text to lowercase
        :param text: input text
        """
        preprocessed_text = text.replace(r"\\u", r"\u").replace("\n", "").encode('utf-8', 'ignore').decode('utf-8').lower()
        return preprocessed_text

    def process_query(self, query):
        merged_doc = self.merge_sections(query)
        return self.tokenize_document(merged_doc)

    def merge_sections(self, sections):
        return ''.join(
            self.preprocess_text(paragraph) for section in sections for paragraph in
            section
        )

    def tokenize_document(self, document):
        tokens = word_tokenize(document)

        tokenized_doc = {
            "string": "",
            "list": []
        }

        for word in tokens:
            if word not in self.stop_words:
                word = self.stemmer.stem(word)
                tokenized_doc["string"] += f" {word}"
                tokenized_doc["list"].append(word)

        return tokenized_doc
