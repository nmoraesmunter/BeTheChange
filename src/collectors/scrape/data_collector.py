import requests
from pymongo import MongoClient
from datetime import datetime
from bs4 import BeautifulSoup
import json
import io


class DataCollector(object):
    def __init__(self, collection_name):

        self.TYPE_USER = 'user'
        self.TYPE_ORG = 'org'
        self.collection_name = collection_name
        # Get MongoDB
        mc = MongoClient()
        self.db = mc['changeorg']
        self.petitions_col = self.db[collection_name]



    def get_htmls(self, petition_url, creator_url, petition_id, creator_type):
        collection = self.db[self.collection_name + "_html"]
        # get petition html
        petition_response = requests.get(petition_url)
        petition_html = petition_response.content
        # get creator html
        creator_response = requests.get(creator_url)
        creator_html = creator_response.content
        # save backup of html in mongodb
        htmls_dict = {'petition_id': petition_id, 'petition_url': petition_url,
                      'creator_url': creator_url, 'petition_html': petition_html,
                      'creator_html': creator_html, 'creator_type': creator_type}
        #collection.insert(htmls_dict)

        return (petition_html, creator_html)

    def get_detailed_data(self, query):
        petitions = self.petitions_col.find(query)
        for petition in petitions:
            petition_id = petition["petition_id"]
            print petition_id
            # Get user
            creator_url = petition["creator_url"]
            org_url = petition["organization_url"]
            creator_type = self.TYPE_USER
            if org_url:
                creator_type = self.TYPE_ORG
                creator_url = org_url
            htmls = self.get_htmls( petition["url"], creator_url, petition_id,creator_type)
            new_fields = self.webscrape(htmls[0], htmls[1], creator_type)
            self.petitions_col.update(
                {'_id': petition['_id']},
                {
                    '$set': new_fields
                }, upsert=True)


    @staticmethod
    def webscrape(petition_html, creator_html, creator_type):
        new_fields = {}
        creator_soup = BeautifulSoup(creator_html, 'html.parser')

        client_data =json.loads(creator_soup.find("script", {"id": "clientData"}).contents[0]) \
                        ["bootstrapData"]["model"]["data"]

        # scrape common attributes user/org
        new_fields["creator_type"] = creator_type
        new_fields["creator_has_website"] = client_data["website"] != None
        new_fields["creator_city"] = client_data["city"]
        try:
            new_fields["creator_photo"] = client_data["photo"]["url"]
        except Exception:
            new_fields["creator_photo"] = None
        new_fields["creator_country"] = client_data["country_code"]
        new_fields["creator_state"] = client_data["state_code"]
        new_fields["creator_has_slug"] = client_data["slug"] != None

        #scrape user
        if creator_type == "user":
            other_petitions = json.loads(creator_soup.find("script", {"id": "clientData"}).contents[0]) \
                ["bootstrapData"]["createdPetitions"]["data"]

            new_fields["creator_first_name"] = client_data["first_name"]
            new_fields["creator_last_name"] = client_data["last_name"]
            new_fields["creator_description"] = client_data["description"]
            new_fields["creator_display_name"] = client_data["display_name"]
            new_fields["creator_locale"] = client_data["locale"]
            new_fields["creator_fb_permissions"] = len(client_data["facebookPermissions"])
        #scrape org
        elif creator_type == "org":
            other_petitions = json.loads(creator_soup.find("script", {"id": "clientData"}).contents[0]) \
                ["bootstrapData"]["petitions"]["data"]

            new_fields["creator_has_address"] = client_data["address"] != None
            new_fields["creator_has_contact_email"] = client_data["admin_contact_email"] != None
            new_fields["creator_has_fb_page"] = client_data["fb_page"] != None
            new_fields["creator_mission"] = client_data["mission"]
            new_fields["creator_org_name"] = client_data["name"]
            new_fields["creator_tax_country_code"] = client_data["tax_country_code"]
            new_fields["creator_tax_state_code"] = client_data["tax_state_code"]
            new_fields["creator_has_twitter"] = client_data["twitter_page"]!= None
            new_fields["creator_has_verified_req"] = client_data["verification_requested_at"]!= None
            new_fields["creator_has_verified_by"] = client_data["verified_by"]!= None
            new_fields["creator_has_verified_at"] = client_data["verified_at"]!= None
            new_fields["creator_has_video"] = client_data["video"]!= None
            new_fields["creator_zipcode"] = client_data["zipcode"]
            new_fields["creator_postal_code"] = client_data["postal_code"]


        #scrape past_petitions stats

        count_victories = 0
        count_verified_victories = 0
        victory_dates = []
        verified_victory_dates = []

        for past_petition in other_petitions:
            count_victories += past_petition["is_victory"]
            count_verified_victories += past_petition["is_verified_victory"]
            victory_date = None
            if past_petition["victory_date"]:
                victory_date = datetime.strptime(past_petition["victory_date"], "%Y-%m-%d")# format YYYY-MM-DD
            if past_petition["is_victory"] and victory_date:
                victory_dates.append(victory_date)
            if past_petition["is_verified_victory"] and victory_date:
                verified_victory_dates.append(victory_date)


        new_fields["num_past_petitions"] = len(other_petitions)
        new_fields["num_past_victories"] = count_victories
        new_fields["num_past_verified_victories"] = count_verified_victories
        if len(victory_dates) > 0:
            new_fields["last_past_victory_date"] = max(victory_dates).strftime("%Y-%m-%d")
        else:
            new_fields["last_past_victory_date"] = None

        if len(verified_victory_dates) > 0:
            new_fields["last_past_verified_victory_date"] = max(verified_victory_dates).strftime("%Y-%m-%d")
        else:
            new_fields["last_past_verified_victory_date"] = None


        #scrape petition
        petition_soup = BeautifulSoup(petition_html, 'html.parser')

        petition_data =json.loads(petition_soup.find("script", {"id": "clientData"}).contents[0]) \
                        ["bootstrapData"]["model"]["data"]

        new_fields["ask"] = petition_data["ask"]
        new_fields["calculated_goal"] = petition_data["calculated_goal"]
        new_fields["description"] = petition_data["description"]
        new_fields["discoverable"] = petition_data["discoverable"]
        new_fields["display_title"] = petition_data["display_title"]
        new_fields["displayed_signature_count"] = petition_data["displayed_signature_count"]
        new_fields["is_pledge"] = petition_data["is_pledge"]
        new_fields["is_victory"] = petition_data["is_victory"]
        new_fields["is_verified_victory"] = petition_data["is_verified_victory"]
        new_fields["languages"] = petition_data["languages"]
        new_fields["original_locale"] = petition_data["original_locale"]
        try:
            new_fields["photo"] =  petition_data["photo"]["url"]
        except Exception:
            new_fields["photo"] = None
        new_fields["progress"] = petition_data["progress"]
        tags = []
        for tag in petition_data["tags"]:
            tags.append(tag["name"])
        new_fields["tags"] = tags
        new_fields["victory_date"] = petition_data["victory_date"]
        new_fields["has_video"] = petition_data["video"]!=None
        new_fields["targets_detailed"] = petition_data["targets"]


        return new_fields





if __name__ == "__main__":
    dc = DataCollector("us_closed_petitions")
    query = {"$and": [ {"targets_detailed": { "$exists": False }}, {"petition_id": { "$nin": [32426, 33751,
                                                                                          36845, 43520,
                                                                                          48535, 63234,
                                                                                          95739, 169422,
                                                                                          248762, 293042,
                                                                                          336425, 401549,
                                                                                          514311, 784585,
                                                                                          828915, 926066,
                                                                                          1029438, 5590214,
                                                                                          5605102, 4126580,
                                                                                              5665790, 1166686,
                                                                                              2701491, 2701621,
                                                                                              4483600]}}]}
    # 32426 failed _id
    # 33751 KeyError: 'model' ["bootstrapData"]["model"]["data"]
    # 36845 CursorNotFound: cursor id '117680278265' not valid at server
    # 43520 CursorNotFound: cursor id '118849559126' not valid at server
    # 48535 CursorNotFound: cursor id '118634304577' not valid at server
    # 63234 CursorNotFound: cursor id '116146162406' not valid at server
    # 95739 CursorNotFound: cursor id '116919800838' not valid at server
    # 169422 CursorNotFound: cursor id '117869467556' not valid at server
    # 248762 cursor id '119465639947' not valid at server
    # 293042 Unterminated string starting at: line 1 column 23746 (char 23745)
    # 336425  cursor id '117042014479' not valid at server
    # 401549 CursorNotFound: cursor id '119010208365' not valid at server
    # 514311 CursorNotFound: cursor id '119833035135' not valid at server
    # 784585 CursorNotFound: cursor id '116375545885' not valid at server
    # 828915 Unterminated string starting at: line 1 column 23746 (char 23745)
    # 926066 Unterminated string starting at: line 1 column 1440 (char 1439)
    # 1029438 .CursorNotFound: cursor id '118484700345' not valid at server
    # 5590214 'NoneType' object has no attribute 'contents'
    # 5605102 'NoneType' object has no attribute 'contents'
    # 4126580 'NoneType' object has no attribute 'contents'
    # 5665790 'NoneType' object has no attribute 'contents'
    # 1166686 KeyError: 'model'
    # 2701491 'NoneType' object has no attribute 'contents'
    # 2701621 ValueError: Unterminated string starting at: line 1 column 23746 (char 23745)
    # 4483600 'NoneType' object has no attribute 'contents'
    print dc.get_detailed_data(query)
