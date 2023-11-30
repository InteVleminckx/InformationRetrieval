from rank_bm25 import BM25Okapi

from src.data_pre_processing import DataPreProcessing


class BM25:

    def __init__(self, preprocessor: DataPreProcessing):
        self.preprocessor: DataPreProcessing = preprocessor
        self.data = preprocessor.data
        self.bm25Okapi = BM25Okapi(preprocessor.tokenized_documents_listed)

    def rank_documents(self, query):
        """
        This function ranks the documents based on the cosine similarity
        :param query: input query
        """

        # Preprocess the query
        query_tokenized = self.preprocessor.process_query(query)

        scores = self.bm25Okapi.get_scores(query_tokenized["list"])

        # Sort the RSV
        mapped = list(zip(self.preprocessor.titles, scores))
        sorted_sim = sorted(mapped, key=lambda x: x[1], reverse=True)

        # Print the top 5 results
        for i in range(5):
            title, score = sorted_sim[i]
            content = self.data[title]
            print(f"{i + 1}) Title: {title} - Score: {score}")
            # print(f"Content: {content}")
