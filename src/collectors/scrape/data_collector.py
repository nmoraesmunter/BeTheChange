import requests
from pymongo import MongoClient, errors
from datetime import datetime
from bs4 import BeautifulSoup
import json
import multiprocessing
from preprocess import text_processor



def get_htmls(petition_url, creator_url, petition_id, creator_type, save = False):
    '''
    Given a petition url and a creator url returns the content of response of a
    requests GET to both urls.
    :param petition_url:
    :param creator_url:
    :param petition_id:
    :param creator_type:
    :param save: if True it saves the raw html with a petition_id in mongo
    :return:
    '''
    collection = db[collection_name + "_html"]
    # get petition html
    petition_response = requests.get(petition_url)
    petition_html = petition_response.content
    # get creator html
    creator_response = requests.get(creator_url)
    creator_html = creator_response.content
    if save:
        # save backup of html in mongodb
        htmls_dict = {'petition_id': petition_id, 'petition_url': petition_url,
                      'creator_url': creator_url, 'petition_html': petition_html,
                      'creator_html': creator_html, 'creator_type': creator_type}
        collection.insert(htmls_dict)

    return petition_html, creator_html


def get_detailed_data(query, webscrape_html = False, get_c = True, get_p = True,
                      get_u = True, get_e = True, get_tf = False):
    '''
    Given a query for the mongo db collection get the flaged information and store it into
    the same document in mongo db. Note that it is an upsert.
    :param query:
    :param webscrape_html:
    :param get_c: get comments?
    :param get_p: get popularity?
    :param get_u: get updates?
    :param get_e: get endorsements?
    :param get_tf: get text features?
    :return: None
    '''
    done = False
    while not done:
        petitions = petitions_col.find(query)
        try:
            for petition in petitions:
                petition_id = petition["petition_id"]
                print petition_id

                new_fields = {}
                if webscrape_html:
                    creator_url = petition["creator_url"]
                    org_url = petition["organization_url"]
                    creator_type = TYPE_USER
                    if org_url:
                        creator_type = TYPE_ORG
                        creator_url = org_url
                    htmls = get_htmls(petition["url"], creator_url, petition_id, creator_type)
                    new_fields = webscrape(htmls[0], htmls[1], creator_type)
                if get_c:
                    new_fields = get_comments(petition_id, new_fields)
                if get_p:
                    new_fields = get_popularity(petition["url"], new_fields)
                if get_u:
                    new_fields = get_updates(petition_id, new_fields)
                if get_e:
                    new_fields = get_endorsements(petition_id, new_fields)
                if get_tf:
                    new_fields = get_text_features(petition_id, new_fields)
                petitions_col.update(
                    {'_id': petition['_id']},
                    {
                        '$set': new_fields
                    }, upsert=True)

            done = True
        except errors.OperationFailure, e:
            msg = e.message
            if not (msg.startswith("cursor id") and msg.endswith("not valid at server")):
                print msg

def get_comments( petition_id, new_fields):
    '''
    Get comments of a petition and store the information in the new_fields dictionary
    :param petition_id:
    :param new_fields:
    :return:
    '''
    last_page = False
    n_items = 0
    idx = 0
    likes = 0
    while not last_page:
        comments_url = "https://www.change.org/api-proxy/-/petitions/%d/comments?limit=10&offset=%d&order_by=voting_score" % (petition_id, idx)
        comments_json = json.loads(requests.get(comments_url).content)
        if "items" in comments_json:
            n_items += len(comments_json["items"])
            for item in comments_json["items"]:
                likes += item["likes"]
            last_page = comments_json["last_page"]
            idx += 10
    new_fields["num_comments"] = n_items
    new_fields["comments_likes"] = likes

    return new_fields

def get_popularity( petition_url, new_fields):
    '''
    Get facebook popularity for the petition url to get the number of shares that are

    number of likes of this URL
    number of shares of this URL (this includes copy/pasting a link back to Facebook)
    number of likes and comments on stories on Facebook about this URL
    number of inbox messages containing this URL as an attachment.

    :param petition_url:
    :param new_fields:
    :return:
    '''
    popular_url = petition_url.replace("api.change.org", "www.change.org")
    fb_api_url = "http://graph.facebook.com/%s"%popular_url


    fb_popularity_json = json.loads(requests.get(fb_api_url).content)

    if "shares" in fb_popularity_json:
        fb_pop = fb_popularity_json["shares"]
    else:
        fb_pop = 0
    new_fields["fb_popularity"] = fb_pop
    return new_fields

def get_updates(petition_id, new_fields):
    updates_url = "https://www.change.org/api-proxy/-/petitions/%d/updates/recent" % petition_id
    updates_json = json.loads(requests.get(updates_url).content)
    num_tweets = 0
    news_coverages = 0
    twitter_popularity = 0
    tweets_followers = 0
    milestones = 0
    last_update = None
    if len(updates_json) > 0:
        last_update = updates_json[0]["created_at"]
    for item in updates_json:
        if item["kind"] == "news_coverage":
            news_coverages += 1
        elif item["kind"] == "verified_tweet":
            num_tweets += 1
            if "embedded_media" in item and "favorite_count" in item["embedded_media"]:
                tweets_followers += item["embedded_media"]["followers_count"]
                twitter_popularity += item["embedded_media"]["favorite_count"] + item["embedded_media"]["retweet_count"]
        else:
            milestones += 1
    new_fields["last_update"] = last_update
    new_fields["num_tweets"] = num_tweets
    new_fields["news_coverages"] = news_coverages
    new_fields["milestones"] = milestones
    new_fields["tweets_followers"] = tweets_followers
    new_fields["twitter_popularity"] = twitter_popularity

    return new_fields


def get_endorsements(petition_id, new_fields):
    endorsements_url = "https://www.change.org/api-proxy/-/petitions/%d/endorsements" % petition_id
    endorsements_json = json.loads(requests.get(endorsements_url).content)
    if "count" in endorsements_json:
        new_fields["endorsements"] = endorsements_json["count"]
    else:
        new_fields["endorsements"] = 0
    return new_fields


def get_text_features(petition, new_fields):
    '''
    Method to get text_features from the petition description and
    save it into the new_fields dictionary.
    :param petition: dictionary
    :param new_fields: dictionary
    :return:
    '''
    descr = petition["description"]
    new_fields["num_capitalized_words_description"] = text_processor.TextProcessor(descr).count_capitalized_words()
    new_fields["num_bold_words_description"] = text_processor.TextProcessor(descr).count_words_bold()
    new_fields["num_italic_words_description"] = text_processor.TextProcessor(descr).count_words_italic()
    new_fields["links_popularity__description"] = text_processor.TextProcessor(descr).get_mean_link_popularity()
    new_fields["num_links__description"] = len(text_processor.TextProcessor(descr).get_links())
    new_fields["has_hashtag_description"] = len(text_processor.TextProcessor(descr).get_hashtags()) > 0

    return new_fields


def webscrape(petition_html, creator_html, creator_type):
    '''
    Webscrapes the petition html and the creator html.

    :param petition_html:
    :param creator_html:
    :param creator_type: user or org
    :return:
    '''
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
    TYPE_USER = 'user'
    TYPE_ORG = 'org'
    collection_name = "us_closed_petitions"
    # Get MongoDB
    mc = MongoClient()
    db = mc['changeorg']
    petitions_col = db[collection_name]

    query1 = {"$and": [ {"endorsements": { "$exists": False }}, {"petition_id": {"$gt":847313 , "$lt": 1100000}}]}
    query2 = {"$and": [ {"endorsements": { "$exists": False }}, {"petition_id": {"$gt":1108398 , "$lt": 1500000}}]}
    query3 = {"$and": [ {"endorsements": { "$exists": False }}, {"petition_id": {"$gt":1500000 , "$lt": 2200000}}]}
    query4 = {"$and": [ {"endorsements": { "$exists": False }}, {"petition_id": {"$gt":2713536 , "$lt": 9000000}}]}

    queries = [query1, query2, query3, query4]

    pool = multiprocessing.Pool(processes=4)
    pool.map(get_detailed_data, queries)


    #dc.get_detailed_data(query)
