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

def get_petitions(start_end):
    id_range = xrange(start_end[0], start_end[1])
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
    # 15/07
    # got from 1087053 to 1132800
    # got from 2587053 to 2648066
    # got from 4087053 to 4340844
    # got from 5587053 to 5647190
    # 16/07
    # got from 1132801 to 1228750
    # got from 2648067 to 2989141
    # got from 4340845 to 4661102
    # got from 5647191 to 6061142
    # 18/07
    # got from 1228750 to 1256188
    # got from 2989141 to 3379465
    # got from 4661102 to 5038746
    # got from 6061142 to 6435755
    #19/07
    # got from 1256188 to
    # got from 3379465 to
    # got from 5038746 to
    # got from 6435755 to 6740252
    #start = xrange(1132801, 6450000 ,1450000)
    start_end = [(1256189,2587052), (3379466, 4087052), (5038747, 5587052), (6435756, 7000000)]
    pool = multiprocessing.Pool(processes=4)
    t = Timer(lambda: pool.map(get_petitions, start_end))
    print "Completed partallel petition crawler in %s seconds." % t.timeit(1)
    print mongo_petitions.count()

    # to get last inserted in mongo
    # db.petitions.find({}).sort({_id:-1}).limit(1)
    # db.petitions.find().sort({petition_id:-1}).limit(1)

    #db.petitions.find({petition_id: {$gt: 1228750 ,$lt: 2587052}}).sort({petition_id:-1}).limit(1)

    #db.petitions.find({petition_id: {$gt: 1228755 ,$lt: 2587052}}).sort({petition_id:1}).limit(1)


