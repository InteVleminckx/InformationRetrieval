from sklearn.metrics.pairwise import cosine_similarity
from src.data_pre_processing import DataPreProcessing
from sentence_transformers import SentenceTransformer


class BERT:
    def __init__(self, preprocessor: DataPreProcessing):
        self.preprocessor = preprocessor
        self.data = preprocessor.data
        self.model = SentenceTransformer(
            "sentence-transformers/bert-base-nli-mean-tokens"
        )
        self.embedded_documents = self.model.encode(
            sentences=preprocessor.tokenized_documents_stringed, show_progress_bar=False
        )

    def rank_documents(self, title: str, k=5):
        title_sentence = self.data[title]["tok-stringed"]
        title_embedding = self.model.encode(
            sentences=title_sentence, show_progress_bar=False
        )

        # Calculate the cosine similarity
        cosine_sim = cosine_similarity([title_embedding], self.embedded_documents)[0]

        # Sort the cosine similarities
        mapped = list(zip(self.preprocessor.titles, cosine_sim))
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
