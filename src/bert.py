from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
from src.data_pre_processing import DataPreProcessing


class BERTSimilarity:
    def __init__(self, preprocessor: DataPreProcessing):
        self.preprocessor: DataPreProcessing = preprocessor
        self.data = preprocessor.data
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.eval()

        # Embed the documents
        self.embedded_data = self.embed_documents(
            preprocessor.tokenized_documents_stringed
        )

    def embed_documents(self, data):
        """
        This function creates a BERT embedding for each document
        :param data: input data
        :return: BERT embeddings for the data
        """
        embedded_data = []
        for doc in data:
            tokens = self.tokenizer(
                doc, return_tensors="pt", truncation=True, padding=True
            )
            with torch.no_grad():
                outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embedded_data.append(embeddings)
        return embedded_data

    def embed_query(self, query):
        """
        This function creates a BERT embedding for the query
        :param query: input query
        :return: BERT embedding for the query
        """
        tokens = self.tokenizer(
            query, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**tokens)
        query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return query_embedding

    def rank_documents(self, query):
        """
        This function ranks the documents based on the cosine similarity
        :param query: input query
        """
        # Preprocess the query
        tokenized_query = self.preprocessor.process_query(query)

        # Embed the query
        query_embedding = self.embed_query(tokenized_query["string"])

        # Calculate the cosine similarity
        similarities = cosine_similarity([query_embedding], self.embedded_data)[0]

        # Sort the cosine similarities
        mapped = list(zip(self.preprocessor.titles, similarities))
        sorted_sim = sorted(mapped, key=lambda x: x[1], reverse=True)

        # Print the top 5 results
        for i in range(5):
            title, score = sorted_sim[i]
            content = self.data[title]
            print(f"{i + 1}) Title: {title} - Score: {score}")
            # print(f"Content: {content}")


# Example usage:
# preprocessor = DataPreProcessing(...)  # Initialize your DataPreProcessing instance
# bert_similarity = BERTSimilarity(preprocessor)
# query = "Your input query here."
# bert_similarity.rank_documents(query)
