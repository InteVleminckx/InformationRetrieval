from src.vector_space_model import VSM

if __name__ == '__main__':
    dataset = "data/video_games.txt"

    vsm = VSM(dataset)
    vsm.read_document()

    vsm.rank_documents("")
