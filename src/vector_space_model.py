import ast

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VSM:

    def __init__(self, document_path):
        self.document_path = document_path
        self.data = {}

        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

        self.tokens = []
        self.titles = []

        self.vectorized_data = None

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

    def preprocess_text(self, text):
        """
        This function preprocesses the text by removing escape characters and converting the text to lowercase
        :param text: input text
        """

        preprocessed_text = text.replace(r"\\u", r"\u").replace("\n", "").encode('utf-8', 'ignore').decode('utf-8').lower()
        return preprocessed_text

    def read_document(self):
        """
        This function reads the document and preprocesses the text
        """

        # Reading the data to a dataframe
        df = pd.read_csv(self.document_path)
        self.titles = df['Title'].tolist()
        # Converting the text to lists
        df["Sections"] = df["Sections"].apply(self.safe_eval)

        # Looping over the rows of the data
        for index, row in df.iterrows():
            title = row['Title']
            sections = row['Sections']

            # Merging the content of the section as one string
            merged_sections = ''.join(
                self.preprocess_text(paragraph) for section in sections for paragraph in
                section
            )

            # Tokenize the merged sections
            tokens = word_tokenize(merged_sections)

            # Remove stop words
            tokenized_doc = ' '.join(word for word in tokens if word not in self.stop_words)
            self.tokens.append(tokenized_doc)
            self.data[title] = merged_sections

        self.vectorized_data = self.create_vector(self.tokens)

    def create_vector(self, data):
        """
        This function creates a vector representation of the data
        :param data: input data
        :return: vector representation of the data
        """
        vectorizer = TfidfVectorizer()
        # parse matrix of (n_samples, n_features) Tf-idf-weighted document-term matrix.
        vector = vectorizer.fit_transform(data)
        return vector

    def rank_documents(self, query):
        """
        This function ranks the documents based on the cosine similarity
        :param query: input query
        """

        # Preprocess the query
        query_tokenized = word_tokenize(query.lower())
        query_preprocced = ' '.join(query_tokenized)

        # Vectorize the documents and the query
        query_vector = self.create_vector([query_preprocced])

        # Calculate the cosine similarity
        cosine_sim = cosine_similarity(query_vector, self.vectorized_data)[0]

        # Sort the cosine similarities

        mapped = list(zip(self.titles, cosine_sim))
        sorted_sim = sorted(mapped, key=lambda x: x[1], reverse=True)

        # Print the top 5 results
        for i in range(5):
            title, score = sorted_sim[i]
            content = self.data[title]
            print(f"{i + 1}) Title: {title} - Score: {score}")
            print(f"Content: {content}")
