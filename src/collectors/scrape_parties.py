import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from src.db.connection import MongoConnection
import multiprocessing
import json


def get_party(slug):
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

def get_slice_parties(sample):
    conn = MongoConnection.default_connection()
    target_parties = conn['changeorg']['target_parties']

    print "Start Get party from petition %d to %d"%(sample[:1]["id"], sample[:-1]["id"])
    sample["party"] = sample.apply(lambda x: get_party(x["slug"]) if x["party"] is None else x, axis=1)


    records = json.loads(sample.T.to_json()).values()
    target_parties.insert(records)
    print "End Get party from petition %d to %d"%(sample[:1]["id"], sample[:-1]["id"])


if __name__ == "__main__":

    #Get MongoDB
    conn = MongoConnection.default_connection()
    petitions_scraped = conn['changeorg']['petitions_scrapped']
    cursor = petitions_scraped.find({"id": {"$gt": 0}})
    df = pd.DataFrame(list(cursor))


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
    list_df.columns = ["name", "position", "state", "id", "description", "slug", "party"]

    procs = 64
    step = list_df.shape[0]/procs

    samples = []
    for start in range(0, list_df.shape[0], step):
        samples.append(list_df[start:start + step])

    pool = multiprocessing.Pool(processes=procs)
    pool.map(get_slice_parties, samples)
    print "finished the process, enjoy your parties!"

