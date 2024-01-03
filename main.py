import multiprocessing
import pprint
import sys
import time

from src.bert import BERT
from src.okapi_BM25 import BM25
from src.vector_space_model import VSM

from src.utils import *

if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn", force=True)

    dataset = "data/video_games.txt"
    # gtl = "data/ground-truth.ground-truth.gt"
    # document_name = "Mafia III"
    #
    # data_preprocessor, renewed = get_preprocessed_data(dataset)
    # ground_truth = get_ground_truth(gtl)
    #
    # if document_name not in data_preprocessor.titles:
    #     print(f"Document '{document_name}' doesn't exists.")
    #     sys.exit(1)
    #
    # print("VSM")
    # vsm = VSM(data_preprocessor, renewed=renewed)
    # result = vsm.rank_documents(document_name, k=15)
    # pprint.pprint(evaluate(result, ground_truth, document_name))
    # #
    # print("\nBM25")
    # bm25 = BM25(data_preprocessor, renewed=renewed)
    # result = bm25.rank_documents(document_name, k=15)
    # pprint.pprint(evaluate(result, ground_truth, document_name))

    # print("\nBERT")
    # start = time.time()
    # bert = BERT(data_preprocessor)
    # bert.parallel_encode_documents(num_processes=2)
    # result = bert.rank_documents(document_name, k=15)
    #
    # averagePrecisionBERT = 0
    # counterBERT = 0
    # total_match = 0
    # for doc in ground_truth[document_name]:
    #     counterBERT += 1
    #     if doc in result:
    #         total_match += 1
    #         averagePrecisionBERT += total_match / (result.index(doc) + 1)
    #
    # recall = (
    #     total_match / sum(len(related_games) for related_games in ground_truth.values())
    #     if len(ground_truth) > 0
    #     else 0
    # )
    # averagePrecisionBERT = averagePrecisionBERT / total_match
    #
    # print(f"BERT precision: {total_match / len(result):.4f}")
    # print(f"BERT recall: {recall:4f}")
    # print(f"BERT Average Precision: {averagePrecisionBERT}")
    # end = time.time() - start
    # print(f"\nElapsed Time: {end // 60:.0f} minutes {end % 60:.2f} seconds")

    import os
    from src.data_pre_processor import DataPreProcessor
    cwd = os.getcwd()
    data_preprocessor = DataPreProcessor(f"{cwd}/{dataset}", cwd)
    vsm = BM25(data_preprocessor)
    result = vsm.rank_documents("Om Nom: Run", k=15)
    print(result)
