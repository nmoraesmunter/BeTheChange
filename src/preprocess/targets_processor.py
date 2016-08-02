from pymongo import MongoClient
import re

class TargetsProcessor(object):
    '''
    Class to process the targets.
    Given a list of targets in json format

    [{
      "id": 761125,
      "display_name": "Kirsten Gillibrand",
      "email": null,
      "type": "Politician",
      "slug": "kirsten-gillibrand",
      "description": "Kirsten Gillibrand is an American politician and the junior United States Senator from New York, in office since 2009. ",
      "publicly_visible": true,
      "verified_at": "2014-02-12T19:53:24Z",
      "summary": "US Senate - New York",
      "locale": "en-US",
      "confirmed_at": "2013-12-12T22:36:20Z",
      "is_person": true,
      "member_of": {},
      "additional_data": {
        "state": "NY",
        "title": "Senator",
        "district": null,
        "active": true
      }, ..]

    We want to compare the data with the congress dataset and get number of democrat targets, number od republican targets and how many responses they did in the past

    '''
    def __init__(self):
        # Get MongoDB
        mc = MongoClient()
        db = mc["changeorg"]
        self.congress = db["congress"]
        self.responses_scraped = db["responses_scraped"]
        self.target_parties = db["target_parties"]


    def get_target_stats(self, targets):


        for target in targets:
            if target["type"] == "Politician":
                self.count_past_responses += self.get_past_responses(target)
                if self.get_party(target) == "D":
                    self.count_democrat_targets += 1
                elif self.get_party(target) == "R":
                    self.count_republican_targets += 1
                elif self.get_party(target) == "Not found":
                    self.count_politician_not_found += 1
            elif target["type"] == "Group":
                self.count_groups += 1
            else:
                self.count_customs +=1


    def get_count_past_responses(self, targets, petition_id):
        count_past_responses = 0
        for target in targets:
            if target["type"] == "Politician":
                count_past_responses += self.get_past_responses(target, petition_id)
        return count_past_responses

    def get_count_democrat_targets(self, targets):
        count_democrat_targets = 0
        for target in targets:
            if target["type"] == "Politician":
                if self.get_party(target) == "D":
                    count_democrat_targets += 1
        return count_democrat_targets

    def get_count_republican_targets(self, targets):
        count_republican_targets = 0
        for target in targets:
            if target["type"] == "Politician":
                if self.get_party(target) == "R":
                    count_republican_targets += 1
        return count_republican_targets

    def get_count_not_found_target(self, targets):
        count_politician_not_found = 0
        for target in targets:
            if target["type"] == "Politician":
                if self.get_party(target) == "Not found":
                    count_politician_not_found += 1
        return count_politician_not_found

    def get_count_groups(self, targets):
        count_groups = 0
        for target in targets:
            if target["type"] == "Group":
                count_groups += 1
        return count_groups

    def get_count_customs(self, targets):
        count_customs = 0
        for target in targets:
            if target["type"] == "Custom":
                count_customs += 1
        return count_customs

    def get_party(self, politician_target):

        cursor = self.target_parties.find({"id":politician_target["id"]})
        full_name = politician_target["display_name"]
        first_name = full_name.split()[0]
        last_name = full_name.split()[-1].upper()
        state_code = politician_target["additional_data"]["state"]
        position = politician_target["additional_data"]["title"]

        if cursor.count() == 0:

            if position.find("Representative") >= 0:
                position = "Representative"
            elif position.find("Senator") >= 0:
                position = "Senator"
            elif position.find("President") >= 0:
                position = "President"

            regx_full_name = re.compile('.*%s(.)*(\\n)?(.)*(%s)(.)*'%(last_name, first_name))
            regx_last_name = re.compile('.*%s(.)*'%last_name)

            query_full_name = {"$and": [
                {"state": state_code},
                {"position": position},
                {"name": {'$regex': regx_full_name}}
            ]}

            query_last_name = {"$and": [
                {"state": state_code},
                {"position": position},
                {"name": {'$regex': regx_last_name}}
            ]}

            if state_code == None:
                state_code = "No state"
                query_full_name = {"$and":[
                        {"position": position},
                        {"name":{'$regex': regx_full_name}}
                        ]}

            cursor = self.congress.find(query_full_name).limit(1)
            if cursor.count() == 0:
                cursor = self.congress.find(query_last_name).limit(1)

        for politician in cursor:
            return politician["party"]

        if cursor.count() == 0:
            print full_name + " " + position + " " + state_code
            return "Not found"

    def get_past_responses(self, politician_target, petition_id):
        cursor = self.responses_scraped.find({"$and": [
            {"petition_id": {"$ne": petition_id}},
            {"decision_maker.id": politician_target["id"]}
        ]})

        return cursor.count()


