import cPickle as pickle
import pandas as pd
from pymongo import MongoClient

def save_pickle(filename, obj):
    with open(filename, 'w') as f:
        f.write(pickle.dumps(obj))

def load_pickle(filename):
    with open(filename, 'r') as f:
        out = pickle.loads(f.read())
    return out

def save_model(model, model_name):
    save_pickle('../app/models/%s.pkl'% model_name, model)

def load_model(model_name):
    model = load_pickle('../app/models/%s.pkl'% model_name)
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