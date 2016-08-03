import cPickle as pickle
import time
import pandas as pd
from src.utils import utils

from flask import Flask, request, render_template
from src.model.model_pipeline import ModelPipeline, ColumnExtractor, ColumnPop, WeightedAdaClassifier, WeightedRFClassifier
from src.model.similarities_pipeline import SimilaritiesPipeline

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
    X, y, petition_id = utils.preprocess_data(project_url)

    y_score = model.pipeline.predict_proba(X)
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

    return render_template('index.html', SUCCESS_SCORE = score, PETITION_ID = petition_id,
                           PREDICTION = prediction, IS_VICTORY = is_victory, PREDICTED = True, SIMILAR = similar_petitions)


if __name__ == '__main__':
    # Load the model
    model = utils.load_model("rf_new_petitions_model")
    similarities_model = utils.load_model("similarities_model")

    app.run(host='0.0.0.0', port=8080, debug=True)
