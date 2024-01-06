def get_ground_truth(path):
    import copy
    import pickle
    with open(path, 'rb') as gt:
        gt_data = gt.read()
        gt = pickle.loads(gt_data)
        gtDC = copy.deepcopy(gt)
        for k in gt.keys():
            for v in gt[k]:
                if v not in gtDC.keys():
                    gtDC[v] = copy.deepcopy(gt[k])
                    gtDC[v].pop(v)
                    gtDC[v][k] = 1
                elif v in gtDC.keys() and k not in gtDC[v].keys():
                    gtDC[v][k] = 1
        return gtDC


def evaluate(result, ground_truth_labels, query_doc):
    avg_precision = 0
    total_match = 0
    results_lower = [res.lower() for res in result]
    for doc in ground_truth_labels[query_doc]:
        if doc.lower() in results_lower:
            total_match += 1
            avg_precision += total_match / (result.index(doc) + 1)

    return {
        "recall": calculate_recall(total_match, ground_truth_labels, query_doc),
        "precision": calculate_precision(total_match, result),
        "AP": calculate_avg_precision(avg_precision, total_match),
    }


def calculate_recall(total_match, gtl, query_doc):
    return round((
        total_match / len(gtl[query_doc]) if len(gtl[query_doc]) > 0 else 0.0
    ), 4)


def calculate_precision(total_match, result):
    return round(total_match / len(result), 4)


def calculate_avg_precision(avg_pre, total_match):
    if total_match == 0:
        return 0.0
    return round(avg_pre / total_match, 4)
