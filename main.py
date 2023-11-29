from src.vector_space_model import VSM
from src.okapi_BM25 import BM25

if __name__ == '__main__':
    dataset = "data/video_games_small.txt"

    vsm = VSM(dataset)
    vsm.read_document()
    vsm.rank_documents("fighting video game")

    bm25 = BM25(dataset)
    bm25.read_document()
    bm25.rank_documents("fighting video game")