from bs4 import BeautifulSoup
import re


class TextProcessor(object):
    def __init__(self, text_xml):
        self.text_xml = text_xml
        self.soup = BeautifulSoup(self.text_xml, "lxml")

    @staticmethod
    def count_words(text):
        try:
            return len(text.split())
        except Exception:
            return 0

    def count_words_bold(self):
        tags = self.soup.find_all('strong')
        num_bold = 0
        for tag in tags:
            num_bold += TextProcessor.count_words(tag.next)
        return num_bold

    def count_words_italic(self):
        tags = self.soup.find_all('em')
        num = 0
        for tag in tags:
            num += TextProcessor.count_words(tag.next)
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


    def get_clean_text(self):
        page_text = ' '.join(self.soup.findAll(text=True))

        return page_text


