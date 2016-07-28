import json
import requests
from bs4 import BeautifulSoup
import re


class TextProcessor(object):
    def __init__(self, text_xml):
        self.text_xml = text_xml
        self.soup = BeautifulSoup(self.text_xml, "lxml")


    def count_words(self, text):
        try:
            return len(text.split())
        except Exception:
            return 0

    def count_words_bold(self):
        tags = self.soup.find_all('strong')
        num_bold = 0
        for tag in tags:
            num_bold += self.count_words(tag.next)
        return num_bold

    def count_words_italic(self):
        tags = self.soup.find_all('em')
        num = 0
        for tag in tags:
            num += self.count_words(tag.next)
        return num

    def count_capitalized_words(self):
        return len(re.findall(r"(\b[A-Z][A-Z0-9]+\b)", self.text_xml))

    def get_hashtags(self):
        return re.findall(r"#(\w+)", self.text_xml)

    def get_links(self):
        tags = self.soup.find_all('a', href=True)
        links = []
        for tag in tags:
            links.append(tag["href"][2:-3])
        return links

    def get_link_popularity(self, url):

        fb_api_url = "http://graph.facebook.com/%s"%url
        '''
        number of likes of this URL
        number of shares of this URL (this includes copy/pasting a link back to Facebook)
        number of likes and comments on stories on Facebook about this URL
        number of inbox messages containing this URL as an attachment.
        '''
        try:
            fb_popularity_json = json.loads(requests.get(fb_api_url).content)

            if "shares" in fb_popularity_json:
                fb_pop = fb_popularity_json["shares"]
            else:
                fb_pop = 0
            return fb_pop
        except:
            return 0

    def get_mean_link_popularity(self):
        links = self.get_links()
        total_popularity = 0
        n = len(links)
        for link in links:
            total_popularity += self.get_link_popularity(link)
        if n > 0:
            return total_popularity/n
        else:
            return 0


