from rank_bm25 import BM25Okapi

from src.data_pre_processor import DataPreProcessor
from src.utils import *


class BM25:

    def __init__(self, preprocessor: DataPreProcessor, renewed=False):
        self.preprocessor: DataPreProcessor = preprocessor
        self.data = preprocessor.preprocessed_data
        self.bm25Okapi = BM25Okapi(self.preprocessor.docs_l_tokenized)

    def rank_documents(self, title, k=5):
        """
        This function ranks the documents based on the cosine similarity
        :param title: the title of the document
        """

        scores = self.bm25Okapi.get_scores(self.data[title]["list"])

        # Sort the RSV
        mapped = list(zip(self.preprocessor.titles, scores))
        sorted_sim = sorted(mapped, key=lambda x: x[1], reverse=True)

        # remove searched query
        sorted_sim = [item for item in sorted_sim if item[0] != title]

        results = [sorted_sim[i][0] for i in range(k)]

        # Print the top k results
        for i in range(k):
            title, score = sorted_sim[i]
            # content = self.data[title]
            print(f"{i + 1}) Title: {title} - Score: {score}")
            # print(f"Content: {content}")

        return results
