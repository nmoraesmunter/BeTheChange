from pymongo import MongoClient

# Get MongoDB
mc = MongoClient()
db = mc['changeorg']
mongo_petitions = db["us_closed_petitions"]
db.drop_collection("petition_ids")
mongo_ids = db["petition_ids"]

cursor = mongo_petitions.find({"petition_id":{"$gt":0}}, {"petition_id":1})


for petition_id in cursor:
    row = {}
    row["id"] = petition_id["petition_id"]
    row["status"] = "new"
    mongo_ids.insert(row)

print "done"