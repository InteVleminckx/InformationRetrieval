from sklearn.metrics.pairwise import cosine_similarity
from src.data_pre_processor import DataPreProcessor
from sentence_transformers import SentenceTransformer
import pickle
import subprocess
import tempfile
import multiprocessing


class BERT:
    def __init__(self, preprocessor: DataPreProcessor):
        self.preprocessor = preprocessor
        self.data = preprocessor.preprocessed_data
        self.num_processes = 4
        self.documents_per_process = (
            len(self.preprocessor.docs_s_tokenized) // self.num_processes
        )
        self.model = SentenceTransformer(
            "sentence-transformers/bert-base-nli-mean-tokens"
        )

    def run_subprocess(self, sublist_file_path, i):
        command = ["python3", "src/sub_program.py", sublist_file_path, str(i)]
        result = subprocess.run(command, capture_output=True, text=True)

        # print subprocess output and error for debugging
        print(f"Subprocess {i} finished with return code {result.returncode}")
        print("Subprocess output:", result.stdout)
        print("Subprocess error:", result.stderr)

    def parallel_encode_documents(self, num_processes=4):
        processes = []

        # Create a temporary directory for sublist files
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(num_processes):
                start_idx = i * self.documents_per_process
                end_idx = (
                    (i + 1) * self.documents_per_process
                    if i != num_processes - 1
                    else None
                )
                sublist = self.preprocessor.docs_s_tokenized[
                    start_idx:end_idx
                ]

                # Save the sublist to a temporary file
                sublist_file_path = f"{temp_dir}/sublist_{i}.pkl"
                with open(sublist_file_path, "wb") as sublist_file:
                    pickle.dump(sublist, sublist_file)

                process = multiprocessing.Process(
                    target=self.run_subprocess,
                    args=(
                        sublist_file_path,
                        i,
                    ),
                )
                processes.append(process)
                process.start()

            # Wait for all processes to finish
            for process in processes:
                process.join()

            embeddings = []
            for i, process in enumerate(processes):
                # Load embeddings from pickle files
                with open(f"embeddings_sublist_{i}.pkl", "rb") as f:
                    embeddings.extend(pickle.load(f))

            # Save the combined embeddings to a single pickle file
            with open("all_embeddings.pkl", "wb") as f:
                pickle.dump(embeddings, f)

            print("All embeddings saved to a single pickle file.")

    def rank_documents(self, title: str, k=5):
        title_sentence = self.data[title]["string"]
        title_embedding = self.model.encode(
            sentences=title_sentence, show_progress_bar=False
        )

        # Load embeddings from the pickle file
        with open("all_embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)

        # Calculate the cosine similarity
        cosine_sim = cosine_similarity([title_embedding], embeddings)[0]

        # Sort the cosine similarities
        mapped = list(zip(self.preprocessor.titles, cosine_sim))
        sorted_sim = sorted(mapped, key=lambda x: x[1], reverse=True)

        # remove searched query
        sorted_sim = [item for item in sorted_sim if item[0] != title]

        results = [sorted_sim[i][0] for i in range(k)]

        # Print the top k results
        for i in range(k):
            title, score = sorted_sim[i]
            print(f"{i + 1}) Title: {title} - Score: {score}")

        return results
