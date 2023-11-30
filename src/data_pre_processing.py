import ast

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class DataPreProcessing:

    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.tokenized_documents_stringed = []
        self.tokenized_documents_listed = []
        self.data = {}
        self.titles = []

    @staticmethod
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

    @staticmethod
    def preprocess_text(text):
        """
        This function preprocesses the text by removing escape characters and converting the text to lowercase
        :param text: input text
        """
        preprocessed_text = text.replace(r"\\u", r"\u").replace("\n", "").encode('utf-8', 'ignore').decode('utf-8').lower()
        return preprocessed_text

    def read_dataset(self, dataset_path):
        """
        This function reads the document and preprocesses the text
        """

        # Reading the data to a dataframe
        df = pd.read_csv(dataset_path)
        self.titles = df["Title"].tolist()

        # Converting the text to lists
        df["Sections"] = df["Sections"].apply(self.safe_eval)

        # Looping over the rows of the data
        for index, row in df.iterrows():
            title = row['Title']
            sections = row['Sections']

            # Merging the content of the section as one string
            merged_sections = self.merge_sections(sections)

            # Tokenize the document and remove the stop words
            tokenized_doc = self.tokenize_document(merged_sections)
            self.tokenized_documents_stringed.append(tokenized_doc["string"])
            self.tokenized_documents_listed.append(tokenized_doc["list"])
            self.data[title] = merged_sections

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
                tokenized_doc["string"] += f" {word}"
                tokenized_doc["list"].append(word)

        return tokenized_doc
