import time

from flask import Flask, request
from flask import render_template, redirect, url_for, jsonify, make_response
from flask import json

from src.bert import BERT
from src.okapi_BM25 import BM25
from src.vector_space_model import VSM

from src.utils import *
import os
import time

app = Flask(__name__)

dataset = "data/video_games_small.txt"
cwd = os.getcwd()
data_preprocessor = DataPreProcessor(f"{cwd}/{dataset}", cwd)
groundTruthLabels = get_ground_truth(f"{cwd}/data/ground-truth.gt")

@app.route('/retrieval', methods = ['POST'])
def statistics():
    global data_preprocessor
    queryTitle = request.get_json()['title']
    topK = int(request.get_json()['topK'])

    vsm = VSM(data_preprocessor)
    bm25 = BM25(data_preprocessor)
    #bert = BERT(data_preprocessor)
    #bert.parallel_encode_documents(num_processes=2)

    startVSM = time.time()
    resultVSM = vsm.rank_documents(queryTitle, k=topK)
    endVSM = time.time()

    startBM = time.time()
    resultBM = bm25.rank_documents(queryTitle, k=topK)
    endBM = time.time()

    startBERT = time.time()
    #resultBERT = bert.rank_documents(queryTitle, k=15)
    endBERT = time.time()

    resultBERT = ["lol", "bitch", "yeet", "jezus", "mozes"]

    return redirect(url_for('retrieved', VSM_res = '#'.join(resultVSM), BM_res = '#'.join(resultBM), BERT_res =
    '#'.join(resultBERT), title = queryTitle, timeVSM = endVSM-startVSM, timeBM = endBM-startBM, timeBERT =
    endBERT-startBERT), code= 302)

@app.route('/retrieved/')
def retrieved():
    global groundTruthLabels
    vms_res = request.args.get('VSM_res').split('#')
    bm_res = request.args.get('BM_res').split('#')
    bert_res = request.args.get('BERT_res').split('#')
    queryTitle = request.args.get('title')


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
        "timeBERT": round(float(request.args.get('timeBERT')), 5)
    }

    return render_template('retrieved.html', data = data)



@app.route('/')
#@app.route('/<documents>')
def index():
    global data_preprocessor
    titles = data_preprocessor.titles
    data_set = data_preprocessor.data_set

    data = {
        "titles": titles,
        "data_set": data_set
    }

    return render_template('index.html', data = data)


if __name__ == "__main__":
    app.run()
