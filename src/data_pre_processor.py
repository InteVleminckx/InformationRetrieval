import ast

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize


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


class DataPreProcessor:
    for data in ['tokenizers/punkt', 'corpora/stopwords']:
        try:
            nltk.data.find(data)
        except:
            nltk.download(data.split('/')[1])

    def __init__(self, dataset_path: str):
        self.df = pd.read_csv(dataset_path)
        self.stemmer = SnowballStemmer('english')
        self.stopwords = set(stopwords.words('english'))

        self.titles = self.df['Title'].to_list()
        self.df['Sections'] = self.df['Sections'].apply(safe_eval)

        self.data_set = {}

        self.preprocessed_data = {}
        self.docs_s_tokenized = []
        self.docs_l_tokenized = []

    def preprocess_data(self):

        for index, row in self.df.iterrows():
            title = row['Title']
            sections = row['Sections']

            document = self.merge_sections_to_document(title, sections)
            doc_clean = self.cleanup_document(document)
            doc_tokenized = self.tokenize_document(doc_clean)

            self.preprocessed_data[title] = doc_tokenized
            self.docs_s_tokenized.append(doc_tokenized['string'])
            self.docs_l_tokenized.append(doc_tokenized['list'])

    def merge_sections_to_document(self, title, sections):
        self.data_set[title] = ''

        for i, section in enumerate(sections):
            header, paragraph = section[0] if i > 0 else title, section[1]
            self.data_set[title] += f"{header}\n{paragraph}"

        return self.data_set[title]

    @staticmethod
    def cleanup_document(document):
        return (document
                .replace(r'\\u', r'\u')
                .replace('\n', ' ')
                .replace(',', '')
                .replace('.', '')
                .encode('utf-8', 'ignore')
                .decode('utf-8')
                .lower()
                )

    def tokenize_document(self, document):

        tokens = word_tokenize(document)

        tokenized_doc = {
            'string': '',
            'list': []
        }

        for word in tokens:
            if word not in self.stopwords:
                word = self.stemmer.stem(word)
                tokenized_doc['string'] += f' {word}'
                tokenized_doc['list'].append(word)

        return tokenized_doc
