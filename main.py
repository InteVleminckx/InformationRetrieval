import pprint
import sys

from src.data_pre_processing import *
from src.okapi_BM25 import BM25
from src.vector_space_model import VSM
from src.bert import BERT
import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    dataset = "data/video_games.txt"

    preprocessed_result = get_preprocessed_data(dataset)
    ground_truth = get_ground_truth("data/ground-truth.gt")
    pprint.pprint(ground_truth)
    data_pre_processor = DataPreProcessing(*preprocessed_result)

    document_name = "Mafia III"

    if document_name not in preprocessed_result[0]:
        print(f"Document '{document_name}' doesn't exists.")
        sys.exit(1)

    print("VSM")
    vsm = VSM(data_pre_processor)
    result = vsm.rank_documents(document_name, k=15)

    averagePrecisionVSM = 0
    total_match = 0
    for doc in ground_truth[document_name]:
        if doc in result:
            total_match += 1
            averagePrecisionVSM += total_match / (result.index(doc) + 1)
    recall = (
        total_match / sum(len(related_games) for related_games in ground_truth.values())
        if len(ground_truth) > 0
        else 0
    )
    averagePrecisionVSM = averagePrecisionVSM / total_match

    print(f"VSM precision: {total_match / len(result):4f}")
    print(f"VSM recall: {recall:4f}")
    print(f"VSM Average Precision: {averagePrecisionVSM}")

    print("\nBM25")
    bm25 = BM25(data_pre_processor)
    result = bm25.rank_documents(document_name, k=15)

    averagePrecisionBM = 0
    total_match = 0
    for doc in ground_truth[document_name]:
        if doc in result:
            total_match += 1
            averagePrecisionBM += total_match / (result.index(doc) + 1)

    recall = (
        total_match / sum(len(related_games) for related_games in ground_truth.values())
        if len(ground_truth) > 0
        else 0
    )
    averagePrecisionBM = averagePrecisionBM / total_match

    print(f"BM25 precision: {total_match / len(result):.4f}")
    print(f"BM25 recall: {recall:4f}")
    print(f"BM25 Average Precision: {averagePrecisionBM}")

    print("\nBERT")
    bert = BERT(data_pre_processor)
    bert.parallel_encode_documents(num_processes=3)
    result = bert.rank_documents(document_name, k=15)

    averagePrecisionBERT = 0
    counterBERT = 0
    total_match = 0
    for doc in ground_truth[document_name]:
        counterBERT += 1
        if doc in result:
            total_match += 1
            averagePrecisionBERT += total_match / (result.index(doc) + 1)

    recall = (
        total_match / sum(len(related_games) for related_games in ground_truth.values())
        if len(ground_truth) > 0
        else 0
    )
    averagePrecisionBERT = averagePrecisionBERT / total_match

    print(f"BERT precision: {total_match / len(result):.4f}")
    print(f"BERT recall: {recall:4f}")
    print(f"BERT Average Precision: {averagePrecisionBERT}")
