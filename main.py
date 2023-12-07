from src.data_pre_processing import get_preprocessed_data, DataPreProcessing
from src.okapi_BM25 import BM25
from src.vector_space_model import VSM
from src.bert import BERT

if __name__ == "__main__":
    dataset = "data/video_games.txt"

    preprocessed_result = get_preprocessed_data(dataset)

    data_pre_processor = DataPreProcessing(*preprocessed_result)

    print("VSM")
    vsm = VSM(data_pre_processor)
    vsm.rank_documents("fighting video game")
    print("\nBM25")
    bm25 = BM25(data_pre_processor)
    bm25.rank_documents("fighting video game")
    print("\nBert")
    bert = BERT(data_pre_processor)
    bert.rank_documents("fighting video game")

