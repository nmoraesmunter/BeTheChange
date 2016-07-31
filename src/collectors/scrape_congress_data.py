import io
from bs4 import BeautifulSoup, Tag
from os import listdir
from pymongo import MongoClient

class CongressDataScrape():

    # info from http://bioguide.congress.gov/biosearch/biosearch.asp


    def __init__(self, path):
        self.path = path

    def get_congress(self, filename, members_list):

        with io.open(filename) as f:
            soup = BeautifulSoup(f, "lxml")

        print "scraping ", filename
        for idx, row in enumerate(soup.find_all('tr')):
            if idx > 1:
                content = row.contents
                member = dict()
                if isinstance(content[1].next.next, Tag):
                    member["name"] = ""
                else:
                    member["name"] = content[1].next.next
                member["position"] = content[5].next
                birthdeathstring = content[3].next
                member["birth"] = birthdeathstring[:birthdeathstring.find("-")]
                member["party"] = content[7].next
                member["state"] = content[9].next
                member["congress"] = content[11].contents[0]
                congress_years = content[11].contents[2][1:-1]
                member["start_year"] = congress_years[:congress_years.find("-")]
                member["end_year"] = congress_years[congress_years.find("-") + 1:]
                members_list.append(member)

        return members_list

    def scrape_data(self):
        members = []
        for f in listdir(self.path):
            members = (self.get_congress(self.path + f, members))

        return members


if __name__ == "__main__":

    #Get MongoDB
    mc = MongoClient()
    db = mc['changeorg']
    db.drop_collection("congress")
    mongo_congress = db["congress"]

    cds = CongressDataScrape("../../data/congress/")
    for doc in cds.scrape_data():
        try:
            mongo_congress.insert(doc)
        except Exception:
            print doc



