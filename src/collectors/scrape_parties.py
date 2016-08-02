from utils.utils import read_mongo
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from src.db.connection import MongoConnection
import json


def getParty(slug):
    url = "https://ballotpedia.org/"
    html = requests.get(url + slug).content
    soup = BeautifulSoup(html, "lxml")
    d = soup.findAll("a", href=True, text='Democratic')
    r = soup.findAll("a", href=True, text='Republican')
    if len(d) > len(r):
        return "D"
    elif len(r) > len(d):
        return "R"
    else:
        return None


if __name__ == "__main__":

    #Get MongoDB
    conn = MongoConnection.default_connection()
    target_parties = conn['changeorg']['target_parties']

    df = read_mongo("changeorg", "petitions_scraped", {"id": {"$gt": 0}})

    list_targets = np.asarray(df["targets"].apply(
        lambda x: [[target["display_name"],
                    target["additional_data"]["title"],
                    target["additional_data"]["state"],
                    target["id"],
                    target["description"],
                    ]
                   for target in x if target["type"] == "Politician"]))

    flat_target_list = []
    for targets in list_targets:
        for target in targets:
            flat_target_list.append(target)
    list_df = pd.DataFrame(flat_target_list)

    list_df["slug"] = list_df[0].apply(lambda x: x.replace(" ", "_"))
    list_df = list_df.drop_duplicates()
    list_df["party"] = list_df.apply(
        lambda x: "D" if x[4] is not None and x[4].lower().find("democrat") >= 0 else None, axis=1)
    list_df["party"] = list_df.apply(
        lambda x: "R" if x[4] is not None and x[4].lower().find("republican") >= 0 else x["party"], axis=1)

    step = list_df.shape[0]/100
    for start in range(0,list_df.shape[0], step):
        print "Start Get party from %d to %d"%(start, start+step)
        sample = list_df[start:start+step]
        sample["party"] = sample.apply(lambda x: getParty(x["slug"]) if x["party"] is None else x, axis=1)
        sample.columns = [ "name", "position", "state", "id", "description","slug", "party"]

        records = json.loads(sample.T.to_json()).values()
        target_parties.insert(records)
        print "End Get party from %d to %d" % (start, start + step)
    print "done"