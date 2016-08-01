from __future__ import division
import json
import requests
import timeit
from bs4 import BeautifulSoup
from datetime import datetime
from src.preprocess.text_processor import TextProcessor
from src.db.connection import MongoConnection
import numpy as np
import multiprocessing


class DataCollector(object):
    def __init__(self, petition_id):
        self.petition_id = petition_id
        self.responses = []
        self.time_petition = 0
        self.time_popularity = 0
        self.time_creator = 0
        self.time_comments = 0
        self.time_updates = 0
        self.time_endorsements = 0
        self.time_responses = 0

    def collect_petition(self):
        petition_url = "https://www.change.org/api-proxy/-/petitions/%d" % self.petition_id
        petition_json = json.loads(requests.get(petition_url).content)

        tp = TextProcessor(petition_json["description"])
        links = tp.get_links()
        total_popularity = 0
        n = len(links)
        pop_mean = 0
        for link in links:
            total_popularity += self.get_fb_popularity(link)
        if n > 0:
            pop_mean = total_popularity / n
        petition_json.update({"links_fb_popularity": pop_mean})

        return petition_json

    def collect_comments(self):
        '''
        Get comments of a petition and store the petition and build a dictionary with the information
        :return: dictionary
        '''
        last_page = False
        n_items = 0
        idx = 0
        likes = 0
        comments_stats = {}
        # We take a maximum of 10 pages (some petitions have more than 20000 and is computationally slow
        for i in xrange(10):
            comments_url = "https://www.change.org/api-proxy/-/petitions/%d/comments?limit=10&offset=%d&order_by=voting_score" \
                           % (self.petition_id, idx)
            comments_json = json.loads(requests.get(comments_url).content)
            if "items" in comments_json:
                n_items += len(comments_json["items"])
                for item in comments_json["items"]:
                    likes += item["likes"]
                last_page = comments_json["last_page"]
                idx += 10
        comments_stats["num_comments"] = n_items
        comments_stats["comments_likes"] = likes
        comments_stats["comments_last_page"] = last_page

        return comments_stats

    def collect_creator_stats(self, is_organization, creator_id):
        creator_stats = {"is_organization": is_organization}
        if is_organization:
            creator_html = requests.get("https://www.change.org/o/%d" % creator_id).content
            creator_soup = BeautifulSoup(creator_html, 'html.parser')
            other_petitions = json.loads(creator_soup.find("script", {"id": "clientData"}).contents[0]) \
                ["bootstrapData"]["petitions"]["data"]
        else:
            creator_html = requests.get("https://www.change.org/u/%d" % creator_id).content
            creator_soup = BeautifulSoup(creator_html, 'html.parser')
            other_petitions = json.loads(creator_soup.find("script", {"id": "clientData"}).contents[0]) \
                ["bootstrapData"]["createdPetitions"]["data"]

        count_past_victories = 0
        count_verified_past_victories = 0
        count_past_petitions = 0
        past_victory_dates = []
        verified_past_victory_dates = []

        for past_petition in other_petitions:
            if past_petition["id"] != self.petition_id:  # We don't count current victory as a past victory or petition
                count_past_petitions += 1
                count_past_victories += past_petition["is_victory"]
                count_verified_past_victories += past_petition["is_verified_victory"]
                victory_date = None
                if past_petition["victory_date"]:
                    victory_date = datetime.strptime(past_petition["victory_date"], "%Y-%m-%d")  # format YYYY-MM-DD
                if past_petition["is_victory"] and victory_date:
                    past_victory_dates.append(victory_date)
                if past_petition["is_verified_victory"] and victory_date:
                    verified_past_victory_dates.append(victory_date)

        creator_stats["num_past_petitions"] = count_past_petitions
        creator_stats["num_past_victories"] = count_past_victories
        creator_stats["num_past_verified_victories"] = count_verified_past_victories

        if len(past_victory_dates) > 0:
            creator_stats["last_past_victory_date"] = max(past_victory_dates).strftime("%Y-%m-%d")
        else:
            creator_stats["last_past_victory_date"] = None

        if len(verified_past_victory_dates) > 0:
            creator_stats["last_past_verified_victory_date"] = max(verified_past_victory_dates).strftime("%Y-%m-%d")
        else:
            creator_stats["last_past_verified_victory_date"] = None

        return creator_stats

    def collect_updates(self):
        updates_url = "https://www.change.org/api-proxy/-/petitions/%d/updates/recent" % self.petition_id
        updates_json = json.loads(requests.get(updates_url).content)
        num_tweets = 0
        news_coverages = 0
        twitter_popularity = 0
        tweets_followers = 0
        milestones = 0
        last_update = None
        updates_stats = {}
        if len(updates_json) > 0:
            last_update = updates_json[0]["created_at"]
        for item in updates_json:
            if item["kind"] == "news_coverage":
                news_coverages += 1
            elif item["kind"] == "verified_tweet":
                num_tweets += 1
                if "embedded_media" in item:
                    if "followers_count" in item["embedded_media"]:
                        tweets_followers += item["embedded_media"]["followers_count"]
                    if "favorite_count" in item["embedded_media"]:
                        twitter_popularity += item["embedded_media"]["favorite_count"] \
                                              + item["embedded_media"]["retweet_count"]
            else:
                milestones += 1
        updates_stats["last_update"] = last_update
        updates_stats["num_tweets"] = num_tweets
        updates_stats["news_coverages"] = news_coverages
        updates_stats["milestones"] = milestones
        updates_stats["tweets_followers"] = tweets_followers
        updates_stats["twitter_popularity"] = twitter_popularity

        return updates_stats

    def collect_endorsements(self):
        endorsements_url = "https://www.change.org/api-proxy/-/petitions/%d/endorsements" % self.petition_id
        endorsements_json = json.loads(requests.get(endorsements_url).content)
        endorsements_stats = {}
        if "count" in endorsements_json:
            endorsements_stats["endorsements"] = endorsements_json["count"]
        else:
            endorsements_stats["endorsements"] = 0
        return endorsements_stats

    def collect_responses(self):
        responses_url = "https://www.change.org/api-proxy/-/petitions/%d/responses" % self.petition_id
        responses_json = json.loads(requests.get(responses_url).content)
        responses_stats = {"num_responses": len(responses_json["items"])}
        self.responses = responses_json["items"]
        return responses_stats

    def get_fb_popularity(self, url):
        '''
        Get facebook popularity for the url to get the number of shares defined as:

        number of likes of this URL
        number of shares of this URL (this includes copy/pasting a link back to Facebook)
        number of likes and comments on stories on Facebook about this URL
        number of inbox messages containing this URL as an attachment.

        :param petition_url:
        :return: fb_pop
        '''
        fixed_url = url
        if not url.startswith("http"):
            fixed_url = "http://%s" % url  # Assume we want http and not https ... TODO Try first with http then https?
        fb_api_url = "http://graph.facebook.com/%s" % fixed_url
        fb_popularity_json = json.loads(requests.get(fb_api_url).content)
        fb_pop = 0
        if "shares" in fb_popularity_json:
            fb_pop = fb_popularity_json["shares"]
        return fb_pop

    def get_detailed_data(self, get_petition=True, get_creator=True, get_comments=True, get_updates=True,
                          get_endorsemements=True, get_fb_popularity=True, get_responses=True):

        detailed_data = {}

        if get_petition:
            t = timeit.Timer(lambda: detailed_data.update(self.collect_petition()))
            self.time_petition += t.timeit(number=1)
        if get_fb_popularity:
            known_url = "http://www.change.org/p/" + detailed_data["slug"]
            t = timeit.Timer(lambda: detailed_data.update({"fb_popularity": self.get_fb_popularity(known_url)}))
            self.time_popularity += t.timeit(number=1)
        if get_creator:
            if "organization" in detailed_data:
                is_org = True
                creator_id = detailed_data["organization"]["id"]
                t = timeit.Timer(lambda: detailed_data.update(self.collect_creator_stats(is_org, creator_id)))
            else:
                is_org = False
                creator_id = detailed_data["user"]["id"]
                t = timeit.Timer(lambda: detailed_data.update(self.collect_creator_stats(is_org, creator_id)))
            self.time_creator += t.timeit(number=1)
        if get_comments:
            t = timeit.Timer(lambda: detailed_data.update(self.collect_comments()))
            self.time_comments += t.timeit(number=1)
        if get_updates:
            t = timeit.Timer(lambda: detailed_data.update(self.collect_updates()))
            self.time_updates += t.timeit(number=1)
        if get_endorsemements:
            t = timeit.Timer(lambda: detailed_data.update(self.collect_endorsements()))
            self.time_endorsements += t.timeit(number=1)
        if get_responses:
            t = timeit.Timer(lambda: detailed_data.update(self.collect_responses()))
            self.time_responses += t.timeit(number=1)
        return detailed_data


def change_petition_id_status(current, collection, status, time=None):
    collection.update({'_id': current['_id']}, {'$set': {'fb_status': status, 'fb_time': time}}, upsert=True)


def all_iteration(start):
    count = 1000
    while count > 0:
        count = one_iteration(start)
        print "[%d] Finished one iteration with count: %d" % (start, count)

    print "[%d] Arrived to the end!"

DB_NAME = 'changeorg'


def one_iteration(start, prefix, limit=1000):
    """
    It will do the following:
        - Find a petition that is status='done' and fb_status='new'
        - Mark this petition_id (or open_petition_id) as 'in_progress'
        - Update the petitions_scrapped with the new fb_popularity field
        - Mark this petition_id (or open_petition_id) as 'done'
    :param start: where to start the filtering to avoid multiple processes doing the same work
    :param prefix: It is a prefix to be used while searching for the collection.
    :param limit: Limit of documents to process in one iteration. It's a desired number.
    :return: the count of petitions still needed to be cleaned
    """
    conn = MongoConnection.default_connection()
    tasks = conn[DB_NAME]['%spetition_ids' % prefix]
    petitions_scrapped = conn[DB_NAME]['%spetitions_scrapped' % prefix]

    to_process = tasks.find({"$and": [{'status': 'done'}, {"fb_status": "new"}, {"id": {"$gt": start}}]}).limit(limit)

    time_popularity = 0
    n = 0

    for current in to_process:
        try:
            print "[%s] Going to process %s" % (datetime.now(), current)
            change_petition_id_status(current, tasks, 'in_progress')
            dc = DataCollector(current['id'])
            url = "https://www.change.org/p/%s" % current['id']
            t = timeit.Timer(lambda: petitions_scrapped.update({'id': current['id']}, {'$set': {"fb_popularity": dc.get_fb_popularity(url)}}, upsert=True))

            time_popularity += t.timeit(number=1)
            change_petition_id_status(current, tasks, 'done', time_popularity)
            n += 1
            if n % 5 == 0:
                print 'scrapped %d petitions, current one %s' % (n, current)
            print "[%s] Done with: %s" % (datetime.now(), petitions_scrapped.findOne({'id': current['id']}))
        except Exception as excp:
            print "[%d] {%s} exception %s" % (start, current, excp)


    if n == 0:
        print "No petition found. Start at %s" % start
    else:
        print "------------------TIMES-----------------------"
        print "Total processed: %s" % n
        print "Popularity : %f" % (time_popularity * 1./ n)

    return tasks.find({"$and": [{'status': 'in_progress'}, {"id": {"$gt": start}}]}).count()


def print_sth(start):
    print start


if __name__ == "__main__":
    procs = 64
    step = 100000
    max_id = 5000000

    starts = range(0, max_id, max_id // procs)
    print "[%s] That's my steps: %s." % (datetime.now(), starts)
    """ TODO uncomment when it works for one case.
    pool = multiprocessing.Pool(processes=procs)
    pool.map(all_iteration, starts)
    """
    one_iteration(1000, "", 1)
    print "[%s] Finished the process, enjoy your scrapped data!" % (datetime.now())
