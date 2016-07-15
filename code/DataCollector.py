import requests
from change_api import ChangeOrgApi
from bs4 import BeautifulSoup

import json
import io



# read API keys
with io.open('api_key.json') as cred:
    creds = json.load(cred)


api = ChangeOrgApi(**creds)

api_url = "https://api.change.org/v1/petitions/"
petitions_url = "https://www.change.org/petitions?hash=recommended&hash_prefix=&list_type=default&view=recommended&page=1"
headers = {'Accept':'application/json, text/javascript, */*',
            'X-Requested-With': 'XMLHttpRequest',
           }


victories_url = "https://www.change.org/victories?first_request=true&hash_prefix=featured&view=featured_victories&page=1"




#TO-DO figure out how to get the json from the app
response = requests.get(victories_url, headers = headers)



# for now read json from file
petitions_data = []
with io.open('petitions_sample.json') as r:
    response = json.load(r)
    soup = BeautifulSoup(response["html"], "lxml")
    petitions = soup.findAll("li", class_ = "petition")
    for petition in petitions:
        p_id = petition.get("data-id")
        api_petition = api.getSinglePetitionById(p_id)
        petition_url = api_petition["url"]
        user_url = api_petition["creator_url"]
        if user_url:
            creator_id = api_petition["creator_url"].replace("https://api.change.org/u/", "")
            if creator_id.isdigit():
                api_user = api.getUserById(creator_id)
                country_code = api_user["country_code"]
                if country_code == "US":
                    api_petition["creator_data"] = api_user
                    detail = requests.get(petition_url)
                    detail_soup = BeautifulSoup(detail.content, 'lxml')
                    detail = detail_soup.select("div.show")
                    api_petition["detail"] = detail
                    petitions_data.append(api_petition)
            else:
                print "User format", creator_id
        else:
            print "Missed petition - no user: ", p_id




print len(petitions_data)








