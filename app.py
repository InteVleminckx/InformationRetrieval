import copy
import os
import time

from flask import Flask, request
from flask import render_template, redirect, url_for

from src.bert import BERT
from src.okapi_BM25 import BM25
from src.utils import *
from src.vector_space_model import VSM

app = Flask(__name__)

dataset = "data/video_games_small.txt"
cwd = os.getcwd()
data_preprocessor = DataPreProcessor(f"{cwd}/{dataset}", cwd)
groundTruthLabels = get_ground_truth(f"{cwd}/data/ground-truth.gt")
vsm = VSM(data_preprocessor)
bm25 = BM25(data_preprocessor)
bert = BERT(data_preprocessor)




@app.route('/retrieval', methods=['POST'])
def statistics():
    global data_preprocessor, vsm, bm25, bert

    queryTitle = request.get_json()['title']
    topK = int(request.get_json()['topK'])

    #running the different retrieval methods on the dataset
    startVSM = time.time()
    resultVSM = vsm.rank_documents(queryTitle, k=topK)
    endVSM = time.time()

    startBM = time.time()
    resultBM = bm25.rank_documents(queryTitle, k=topK)
    endBM = time.time()

    startBERT = time.time()
    resultBERT = bert.rank_documents(dataset, queryTitle, k=topK)
    endBERT = time.time()

    #redirecting to retrieved page
    return redirect(url_for('retrieved', VSM_res='#'.join(resultVSM), BM_res='#'.join(resultBM), BERT_res=
    '#'.join(resultBERT), title=queryTitle, timeVSM=endVSM - startVSM, timeBM=endBM - startBM, timeBERT=
                            endBERT - startBERT), code=302)


@app.route('/retrieved/')
def retrieved():
    global groundTruthLabels
    data_set = data_preprocessor.data_set

    vms_res = request.args.get('VSM_res').split('#')
    bm_res = request.args.get('BM_res').split('#')
    bert_res = request.args.get('BERT_res').split('#')
    queryTitle = request.args.get('title')

    #calculating the evaluation metrics on the retrieved results
    evalVSM = evaluate(vms_res, groundTruthLabels, queryTitle)
    evalBM25 = evaluate(bm_res, groundTruthLabels, queryTitle)
    evalBERT = evaluate(bert_res, groundTruthLabels, queryTitle)

    data = {
        "vms_res": vms_res,
        "bm_res": bm_res,
        "bert_res": bert_res,
        "vsm_eval": evalVSM,
        "bm25_eval": evalBM25,
        "bert_eval": evalBERT,
        "title": queryTitle,
        "timeVSM": round(float(request.args.get('timeVSM')), 5),
        "timeBM": round(float(request.args.get('timeBM')), 5),
        "timeBERT": round(float(request.args.get('timeBERT')), 5),
        "data_set": data_set
    }

    return render_template('retrieved.html', data=data)


@app.route('/')
def index():
    global data_preprocessor
    titles = copy.deepcopy(data_preprocessor.titles)
    data_set = data_preprocessor.data_set
    titles.sort()   # sorting the titles
    data = {
        "titles": titles,
        "data_set": data_set
    }

    return render_template('index.html', data=data)


if __name__ == "__main__":
    app.run()
