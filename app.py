from flask import Flask
from flask import render_template
from flask import json

from src.utils import *

app = Flask(__name__)

data_preprocessor = get_preprocessed_data("video_games.txt")


@app.route('/')
@app.route('/<documents>')
def index():
    global data_preprocessor
    titles = data_preprocessor[0].titles
    #print(titles, flush=True)

    data = {
        "titles": titles
    }

    return render_template('index.html', data = data)


if __name__ == "__main__":
    app.run()
