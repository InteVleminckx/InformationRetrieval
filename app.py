from flask import Flask
from flask import render_template

from src.utils import *

app = Flask(__name__)

data_preprocessor = get_preprocessed_data("video_games.txt")


@app.route('/')
def index():
    global hello
    print(hello, flush=True)
    return render_template('index.html')


if __name__ == "__main__":
    app.run()
