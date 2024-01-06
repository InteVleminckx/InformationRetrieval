import os
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_pre_processor import DataPreProcessor


class BERT:
    def __init__(self, preprocessor: DataPreProcessor):
        self.preprocessor = preprocessor
        self.data = preprocessor.preprocessed_data
        self.model = SentenceTransformer(
            "sentence-transformers/bert-base-nli-mean-tokens"
        )

    def rank_documents(self, dataset_title: str, query: str, k=5) -> list:
        """
        Rank the documents based on the cosine similarity between the query and the documents
        :param dataset_title: title of the dataset (this is usually the path to the file containing the dataset)
        :param query: query to rank the documents for
        :param k: number of documents to return
        """
        # Create the embedding for the title
        query_lower = query.lower()
        title_sentence = self.data[query_lower]["string"]
        title_embedding = self.model.encode(
            sentences=title_sentence, show_progress_bar=False
        )

        # Check if the file with the name dataset_title.pkl exists. If it does not exist, create it
        if not os.path.exists(f"{dataset_title}.pkl"):
            # Create the embeddings for all documents
            document_embeddings = self.model.encode(sentences=self.preprocessor.docs_s_tokenized,
                                                    show_progress_bar=False)
            with open(f"{dataset_title}.pkl", "wb") as f:
                pickle.dump(document_embeddings, f)
        else:
            # Load the embeddings from the pickle file
            with open(f"{dataset_title}.pkl", "rb") as f:
                document_embeddings = pickle.load(f)

        # Calculate the cosine similarity
        cosine_sim = cosine_similarity([title_embedding], document_embeddings)[0]

        # Sort the cosine similarities
        mapped = list(zip(self.preprocessor.titles, cosine_sim))
        sorted_sim = sorted(mapped, key=lambda x: x[1], reverse=True)

        # remove searched query
        sorted_sim = [item for item in sorted_sim if item[0] != query]
        results = [sorted_sim[i][0] for i in range(k)]

        return results
