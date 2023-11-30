from src.data_pre_processing import DataPreProcessing
from src.okapi_BM25 import BM25
from src.vector_space_model import VSM

if __name__ == '__main__':
    dataset = "data/video_games.txt"

    data_pre_processor = DataPreProcessing()
    data_pre_processor.read_dataset(dataset)

    print("VSM")
    vsm = VSM(data_pre_processor)
    vsm.rank_documents("fighting video game")
    print("\nBM25")
    bm25 = BM25(data_pre_processor)
    bm25.rank_documents("fighting video game")
