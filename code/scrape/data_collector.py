import requests
from pymongo import MongoClient
from bs4 import BeautifulSoup
import json
import io



class DataCollector(object):

    def __init__(self):

        # Get MongoDB
        mc = MongoClient()
        self.db = mc['changeorg']
        self.mongo_petitions_html = self.db["petitions_html"]
        self.mongo_users = self.db["users"]
        self.mongo_organizations = self.db["organizations"]


    def webscrape_petition(self, url, petition_id):
        response = requests.get(url)
        html = response.content
        htmls_dict = {'petition_id': petition_id, 'html': html}
        self.mongo_petitions_html.insert(htmls_dict)

    def get_creator(self, url, petition_id):
        response = requests.get(url)
        html = response.content
        htmls_dict = {'petition_id': petition_id, 'html': html}
        self.mongo_users.insert(htmls_dict)



if __name__ == "__main__":

    mc = MongoClient()
    db = mc['changeorg']
    mongo_petitions= db["petitions"]


    petitions = mongo_petitions.find({ "$or": [{"status": "closed"}, {"status": "victory"}]}).limit(2)
    dc = DataCollector()
    for petition in petitions:
        petition_id = petition["petition_id"]
        #Get html
        dc.webscrape_petition(petition["url"], petition_id)
        # Get user
        creator_url = petition["creator_url"]
        dc.get_creator(creator_url, petition_id)

        #http://stackoverflow.com/questions/24118337/fetch-data-of-variables-inside-script-tag-in-python-or-content-added-from-js


