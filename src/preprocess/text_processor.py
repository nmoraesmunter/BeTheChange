import json
import requests
from bs4 import BeautifulSoup


class TextProcessing(object):
    def __init__(self, text_xml):
        self.text_xml = text_xml
        self.soup = BeautifulSoup(self.text_xml)


    def count_words(self, text):
        return len(text.split())

    def count_words_bold(self):
        tags = self.soup.find_all('strong')
        num_bold = 0
        for tag in tags:
            num_bold += self.count_words(tag)
        return num_bold

    def count_words_italic(self):
        "<em>Billions of pounds</em>"
        pass

    def count_capitalized_words(self):
        pass

    def get_hashtag(self):
        "#WhatTheFork"
        pass

    def get_links(self):
       " < ahref =\"http://www.uglyfruitandveg.org/\"rel =\"nofollow\">UglyFruitandVeg.org< / a >"

    def get_link_popularity(url):

        fb_api_url = "http://graph.facebook.com/%s"%url
        '''
        number of likes of this URL
        number of shares of this URL (this includes copy/pasting a link back to Facebook)
        number of likes and comments on stories on Facebook about this URL
        number of inbox messages containing this URL as an attachment.
        '''

        fb_popularity_json = json.loads(requests.get(fb_api_url).content)

        if "shares" in fb_popularity_json:
            fb_pop = fb_popularity_json["shares"]
        else:
            fb_pop = 0
        return fb_pop




