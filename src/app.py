import copy
import os
import time

from flask import Flask, request
from flask import render_template, redirect, url_for

from src.algorithms.bert import BERT
from src.algorithms.okapi_BM25 import BM25
from src.algorithms.vector_space_model import VSM
from src.data_pre_processor import DataPreProcessor
from src.utils import *

app = Flask(__name__)

dataset = "data/video_games.txt"
cwd = os.getcwd()
data_preprocessor = DataPreProcessor(f"{cwd}/{dataset}", cwd)
ground_truth_labels = get_ground_truth(f"{cwd}/data/ground-truth.gt")
vsm = VSM(data_preprocessor)
bm25 = BM25(data_preprocessor)
bert = BERT(data_preprocessor)


@app.route('/retrieval', methods=['POST'])
def statistics():
    """

    :return:
    """
    global data_preprocessor, vsm, bm25, bert

    query_title = request.get_json()['title']
    top_k = int(request.get_json()['topK'])

    # running the different retrieval methods on the dataset
    start_vsm = time.time()
    result_vsm = vsm.rank_documents(query_title, k=top_k)
    end_vsm = time.time()

    start_bm = time.time()
    result_bm = bm25.rank_documents(query_title, k=top_k)
    end_bm = time.time()

    start_bert = time.time()
    result_bert = bert.rank_documents(dataset, query_title, k=top_k)
    end_bert = time.time()

    # redirecting to retrieved page
    return redirect(url_for('retrieved', VSM_res='#'.join(result_vsm), BM_res='#'.join(result_bm), BERT_res=
    '#'.join(result_bert), title=query_title, timeVSM=end_vsm - start_vsm, timeBM=end_bm - start_bm, timeBERT=
                            end_bert - start_bert), code=302)


@app.route('/retrieved/')
def retrieved():
    """

    :return:
    """
    global ground_truth_labels
    data_set = data_preprocessor.data_set

    vms_res = request.args.get('VSM_res').split('#')
    bm_res = request.args.get('BM_res').split('#')
    bert_res = request.args.get('BERT_res').split('#')
    query_title = request.args.get('title')

    # calculating the evaluation metrics on the retrieved results
    eval_vsm = evaluate(vms_res, ground_truth_labels, query_title)
    eval_bm25 = evaluate(bm_res, ground_truth_labels, query_title)
    eval_bert = evaluate(bert_res, ground_truth_labels, query_title)

    data = {
        "vms_res": vms_res,
        "bm_res": bm_res,
        "bert_res": bert_res,
        "vsm_eval": eval_vsm,
        "bm25_eval": eval_bm25,
        "bert_eval": eval_bert,
        "title": query_title,
        "timeVSM": round(float(request.args.get('timeVSM')), 5),
        "timeBM": round(float(request.args.get('timeBM')), 5),
        "timeBERT": round(float(request.args.get('timeBERT')), 5),
        "data_set": data_set,
        "gt": ground_truth_labels
    }

    return render_template('retrieved.html', data=data)


@app.route('/')
def index():
    """

    :return:
    """
    global data_preprocessor
    titles = copy.deepcopy(data_preprocessor.titles)
    data_set = data_preprocessor.data_set
    titles.sort()  # sorting the titles
    data = {
        "titles": titles,
        "data_set": data_set
    }

    return render_template('index.html', data=data)


if __name__ == "__main__":
    app.run()
