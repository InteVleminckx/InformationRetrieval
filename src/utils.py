import pickle

from src.data_pre_processor import DataPreProcessor


def save_to_pickle(file_name, content):
    try:
        with open(f'saved_data/{file_name}.pkl', 'wb') as plk_file:
            pickle.dump(content, plk_file)
        return True
    except:
        return False


def load_from_pickle(file_name):
    try:
        with open(f'saved_data/{file_name}.pkl', 'rb') as plk_file:
            return pickle.load(plk_file)

    except:
        return None


def get_preprocessed_data(dataset):
    dataset_name = dataset.split('.')[:-1][-1].split('/')[-1]
    data_preprocessor = load_from_pickle(f'preprocessed_data_{dataset_name}')
    renewed = False
    if not data_preprocessor:
        renewed = True
        data_preprocessor = DataPreProcessor(f'data/{dataset}')
        data_preprocessor.preprocess_data()
        save_to_pickle(f'preprocessed_data_{dataset_name}', data_preprocessor)

    return data_preprocessor, renewed


def get_ground_truth(path):
    with open(path, 'rb') as gt:
        gt_data = gt.read()
        return pickle.loads(gt_data)


def evaluate(result, ground_truth_labels, query_doc):
    avg_precision = 0
    total_match = 0
    for doc in ground_truth_labels[query_doc]:
        if doc in result:
            total_match += 1
            avg_precision += total_match / (result.index(doc) + 1)

    return {
        "recall": calculate_recall(total_match, ground_truth_labels),
        "precision": calculate_precision(total_match, result),
        "AP": calculate_avg_precision(avg_precision, total_match)
    }


def calculate_recall(total_match, gtl):
    return round((
        total_match / sum(len(related_games) for related_games in gtl.values())
        if len(gtl) > 0 else 0
    ), 4)


def calculate_precision(total_match, result):
    return round(total_match / len(result), 4)


def calculate_avg_precision(avg_pre, total_match):
    return round(avg_pre / total_match, 4)
