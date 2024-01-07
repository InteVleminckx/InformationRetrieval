from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_pre_processor import DataPreProcessor


class VSM:

    def __init__(self, preprocessor: DataPreProcessor):
        self.preprocessor: DataPreProcessor = preprocessor
        self.data = preprocessor.preprocessed_data
        self.tf_idf_vectorizer = TfidfVectorizer()
        self.vectorized_data = self.create_vector_documents(self.preprocessor.docs_s_tokenized)

    def create_vector_documents(self, data):
        """
        This function creates a vector representation of the data
        :param data: input data
        :return: vector representation of the data
        """
        # parse matrix of (n_samples, n_features) Tf-idf-weighted document-term matrix.
        return self.tf_idf_vectorizer.fit_transform(data)

    def create_vector_query(self, query):
        # Uses the vocabulary and document frequencies (df) learned by fit fit_transform.
        return self.tf_idf_vectorizer.transform(query)

    def rank_documents(self, title, k=5):
        """
        This function ranks the documents based on the cosine similarity
        :param title: the title of the document
        :param k: the length of the top results list.
        """

        lowered_title = title.lower()

        # Vectorize the document of the given title
        query_vector = self.create_vector_query([self.data[lowered_title]["string"]])

        # Calculate the cosine similarity
        cosine_sim = cosine_similarity(query_vector, self.vectorized_data)[0]

        # Sort the cosine similarities
        mapped = list(zip(self.preprocessor.titles, cosine_sim))
        sorted_sim = sorted(mapped, key=lambda x: x[1], reverse=True)

        # remove searched query
        sorted_sim = [item for item in sorted_sim if item[0] != title]

        return [sorted_sim[i][0] for i in range(min(k, len(sorted_sim)))]
