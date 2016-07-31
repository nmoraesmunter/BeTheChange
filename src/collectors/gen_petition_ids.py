from pymongo import MongoClient

# Get MongoDB
mc = MongoClient()
db = mc['changeorg']
mongo_petitions = db["petitions"]
db.drop_collection("open_petition_ids")
mongo_ids = db["open_petition_ids"]

cursor = mongo_petitions.find({"$and": [{ "status": { "$nin": ['preview','closed', 'victory'] }}, { "targets.type": 'us_government' }]}, {"petition_id":1})


for petition_id in cursor:
    row = {}
    row["id"] = petition_id["petition_id"]
    row["status"] = "new"
    mongo_ids.insert(row)

print "done"