import pandas as pd
import ast
import nltk
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

        preprocessed_text = text.replace(r"\\u", r"\u").encode('utf-8', 'ignore').decode('utf-8').lower()
        return preprocessed_text

    def read_document(self):
        """
        This function reads the document and preprocesses the text
        """

        # Reading the data to a dataframe
        df = pd.read_csv(self.document_path)

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
            tokenized_doc = ' '.join(word for word in tokens if word.lower() not in self.stop_words)
            self.tokens.append(tokenized_doc)
            self.data[title] = merged_sections

    def rank_documents(self, query):
        """
        This function ranks the documents based on the cosine similarity
        :param query: input query
        """

        # Preprocess the query
        query_tokenized = word_tokenize(query.lower())
        query_preprocced = ' '.join(query_tokenized)

        # Vectorize the documents and the query
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_data = tfidf_vectorizer.fit_transform(self.tokens)
        query_vector = tfidf_vectorizer.transform([query_preprocced])

        # Calculate the cosine similarity
        cosine_sim = cosine_similarity(query_vector, tfidf_data)[0]

        # Sort the cosine similarities
        titles = [key for key in self.data.keys()]
        mapped = list(zip(titles, cosine_sim))
        sorted_sim = sorted(mapped, key=lambda x: x[1], reverse=True)

        # Print the top 5 results
        for i in range(5):
            title, score = sorted_sim[i]
            content = self.data[title]
            print(f"Title: {title} - Score: {score}")
            print(f"Content: {content}")
