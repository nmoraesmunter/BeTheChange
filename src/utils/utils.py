import cPickle as pickle
from pymongo import MongoClient
from src.db.connection import MongoConnection
import requests
from bs4 import BeautifulSoup
import json
from src.collectors.data_collector import DataCollector
from src.preprocess.data_pipeline import DataPipeline
import pandas as pd

def drop_columns(df, columns):
    for col in columns:
        if col in df:
            df.pop(col)
    return df

def preprocess_data(url, is_petitioner):
    conn = MongoConnection.default_connection()
    collection  = 'featured_petitions'
    if is_petitioner:
        collection = 'featured_petitions_raw'

    featured_petitions = conn['changeorg'][collection]

    html = requests.get(url).content
    petition_soup = BeautifulSoup(html, 'html.parser')

    petition_id = json.loads(petition_soup.find("script", {"id": "clientData"}).contents[0]) \
        ["bootstrapData"]["model"]["data"]["id"]

    cursor = featured_petitions.find({"id": petition_id})

    if cursor.count() == 0:
        dc = DataCollector(petition_id)
        petition_json = dc.get_detailed_data()
        df = pd.DataFrame.from_dict(petition_json, orient= "index").T
        dp = DataPipeline(df, is_petitioner, True)
        final_df = dp.apply_pipeline()
    else:
        final_df = pd.DataFrame.from_dict(list(cursor))

    to_drop = ["display_title", "letter_body", "_id", "is_verified_victory", "id"]
    if is_petitioner:
        to_drop = ["display_title", "letter_body","_id", "is_verified_victory", "id",
               "displayed_signature_count", "displayed_supporter_count"]
    final_df = drop_columns(final_df, to_drop)
    y = final_df.pop("status")

    X = final_df

    return X,y, petition_id

def save_pickle(filename, obj):
    with open(filename, 'w') as f:
        f.write(pickle.dumps(obj))

def load_pickle(filename):
    with open(filename, 'r') as f:
        out = pickle.loads(f.read())
    return out

def save_model(model, model_name):
    save_pickle('src/app/models/%s.pkl'% model_name, model)

def load_model(model_name):
    model = load_pickle('src/app/models/%s.pkl'% model_name)
    return model


def read_mongo(database_name, collection_name, query={}, no_id=False):
    '''
        Read from Mongo and put it into aDataFrame
    :param database_name:
    :param collection_name:
    :param query:
    :param no_id:
    :return:
    '''

    # Get MongoDB
    mc = MongoClient()
    db = mc[database_name]
    petitions_col = db[collection_name]

    # Make a query to the specific DB and Collection
    cursor = db[collection_name].find(query)

    # Expand the cursor and construct the DataFrame
    df = pd.DataFrame(list(cursor))

    # Delete the _id
    if no_id:
        del df['_id']

    return df