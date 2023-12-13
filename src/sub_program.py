# sub_program.py

from sentence_transformers import SentenceTransformer
import sys
import pickle


def encode_documents(sublist_file_path):
    with open(sublist_file_path, "rb") as f:
        sublist = pickle.load(f)

    model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")
    embeddings = model.encode(sublist, show_progress_bar=False)
    return embeddings


if __name__ == "__main__":
    sublist_file_path = sys.argv[1]
    sublist_index = int(sys.argv[2])
    embeddings = encode_documents(sublist_file_path)

    # Save embeddings to pickle file
    with open(f"embeddings_sublist_{sublist_index}.pkl", "wb") as f:
        pickle.dump(embeddings, f)
