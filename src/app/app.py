import cPickle as pickle
import time
import pandas as pd
from src.utils import utils

from flask import Flask, request, render_template
from src.model.model_pipeline import ModelPipeline, ColumnExtractor, ColumnPop, WeightedAdaClassifier, \
    WeightedRFClassifier, WeightedSVM
from src.model.similarities_pipeline import SimilaritiesPipeline
import sys
sys.settrace

app = Flask(__name__)


# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Dashboard page
@app.route('/predict', methods=['POST'])
def predict():
    # Scrape urls
    project_url = str(request.form['petition_url'])
    is_petitioner = False
    X, y, petition_id = utils.preprocess_data(project_url, is_petitioner)

    if is_petitioner:
        y_score = model_petitioner.pipeline.predict_proba(X)
    else:
        y_score = model_user.pipeline.predict_proba(X)
    '''
    Project Success Score
    '''
    score = round(y_score[0][1] * 100, 2)
    prediction = "Oh oh, it looks bad...   "
    is_victory = 0
    if score > 50:
        prediction = "This is going to change the world!  "
        is_victory = 1

    '''
    Get similar petitions
    '''
    similar_petitions = similarities_model.top_similar_petitions(X["description"][0])
  #  similar_petitions = [1423479, 808920, 1423479, 908914]

    return render_template('index.html', SUCCESS_SCORE = score, PETITION_ID = petition_id,
                           PREDICTION = prediction, IS_VICTORY = is_victory,
                           PREDICTED = True, SIMILAR=similar_petitions)


if __name__ == '__main__':
    # Load the model
    model_petitioner = utils.load_model("rf_model_petitions")
    model_user = utils.load_model("rf_model_petitions")
    similarities_model = utils.load_model("similarities_model")

    app.run(host='0.0.0.0', port=5353, debug=False, threaded=True)
