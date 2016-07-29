import cPickle as pickle
import time
import pandas as pd


from flask import Flask, request, render_template

app = Flask(__name__)


# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Dashboard page
@app.route('/dashboard', methods=['POST'])
def predict():

     # Scrape urls
    project_url = str(request.form['petition_url'])



    '''
    Project Success Score
    '''
    score = 85



    return render_template('dashboard.html', PROJECT_URL=project_url,  SUCCESS_SCORE = score)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
