from change_api import ChangeOrgApi
from pymongo import MongoClient
import multiprocessing
import json
import io


def get_petitions(start):
    id_range = xrange(start, start + 1750000)
    for id in id_range:
        try:
            petition = api.getSinglePetitionById(id)
            mongo_petitions.insert(petition)
        except Exception:
            print "ID:", id, " doesn't exist"


if __name__ == '__main__':
    # read API keys
    with io.open('api_key.json') as cred:
        creds = json.load(cred)

    api = ChangeOrgApi(**creds)

    #Get MongoDB
    mc = MongoClient()
    db = mc['changeorg']
    mongo_petitions = db["petitions"]

    #Get the petitions using 4 processes
    start = xrange(0, 7000000, 1750000)
    pool = multiprocessing.Pool(processes=4)
    pool.map(get_petitions, start)

    print mongo_petitions.count()



