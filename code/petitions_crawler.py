from change_api import ChangeOrgApi
from pymongo import MongoClient
from timeit import Timer
import multiprocessing
import json
import io


#35859 --> 2010
#for id in xrange(6816731, 6914304, 3):
#for id in xrange(30200, 6915000, 1): stopped at 949716
# for id in xrange(949717, 1949717, 1):

def get_petitions(start):
    id_range = xrange(start, start + 1500000)
    for id in id_range:
        try:
            petition = api.getSinglePetitionById(id)
            mongo_petitions.insert(petition)
        except Exception:
            print "Failed:", id


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
    # I start at id 1087053 because I already downloaded the rest with a single process
    start = xrange(1087053, 6450000 ,1500000)
    pool = multiprocessing.Pool(processes=4)
    t = Timer(lambda: pool.map(get_petitions, start))
    print "Completed partallel petition crawler in %s seconds." % t.timeit(1)
    print mongo_petitions.count()

    # to get last inserted in mongo
    # db.market.find({}).sort({_id:-1}).limit(1)


