from __future__ import division
import numpy as np
import pandas as pd
import text_processor
from targets_processor import TargetsProcessor
from src.db.connection import MongoConnection
import json


class DataPipeline(object):

    def __init__(self, df, raw, to_predict):
        self.df = df
        self.raw = raw
        # Including features of petition performance over time, like social media popularity or signature count
        all_features = ["calculated_goal", "comments_likes",
                        "endorsements",
                        "fb_popularity", "links_fb_popularity", "milestones", "news_coverages",
                        "num_comments", "num_past_petitions", "num_past_verified_victories",
                        "num_past_victories", "num_responses", "num_tweets",
                        "status", "twitter_popularity", "tweets_followers", "description",
                        "display_title", "letter_body", "id",
                        "is_en_US", "is_organization", "is_pledge",
                        "comments_last_page", "is_verified_victory",
                        "goal_days_ratio", "is_en", "progress"]



        # Including just the raw features od a petition, the ones that the petition has in his original state
        raw_features = ["calculated_goal",
                        "num_past_petitions", "num_past_verified_victories",
                        "num_past_victories",
                        "status", "description",
                        "display_title", "letter_body", "id",
                        "is_en_US", "is_organization", "is_pledge", "goal_days_ratio",
                        "is_verified_victory",
                        "is_en"]

        self.featured_columns = all_features
        if raw:
            self.featured_columns = raw_features

        self.target = "status"
        self.to_predict = to_predict


    def drop_columns(self, to_drop):
        for col in to_drop:
            if col in self.df:
                self.df.pop(col)

    def to_datetime(self, to_date_time):
        for col in to_date_time:
            self.df[col] = pd.to_datetime(self.df[col])

    def fill_nan_with_zeros(self, to_fill_with_zeros):
        for col in to_fill_with_zeros:
            self.df[col].fillna(0, inplace=True)
            self.featured_columns += [col]

    def clean_data(self):

        to_drop = ["goal", "calculated_goal_with_endorsers", "creator_name", "title", "document_id",
                   "endorser_count", "organization", "petition_title", "photo_id", "primary_target",
                   "targeting_description", "total_signature_count", "total_supporter_count",
                   "victory_date", "victory_description", "weekly_signature_count"]
        to_date_time = ["created_at", "end_date", "last_past_verified_victory_date",
                        "last_past_victory_date", "last_update"]
        to_fill_with_zeros = ["displayed_signature_count", "displayed_supporter_count"]

        self.drop_columns(to_drop)
        self.to_datetime(to_date_time)
        self.fill_nan_with_zeros(to_fill_with_zeros)

    def generate_text_features(self, columns):

        text_features_df = pd.DataFrame()
        has_cols = []
        for column in columns:
            text_features_df["num_capitalized_words_" + column] = self.df[column]. \
                apply(lambda x: text_processor.TextProcessor(x).count_capitalized_words())
            text_features_df["num_bold_words_" + column] = self.df[column]. \
                apply(lambda x: text_processor.TextProcessor(x).count_words_bold())
            text_features_df["num_italic_words_" + column] = self.df[column]. \
                apply(lambda x: text_processor.TextProcessor(x).count_words_italic())
            text_features_df["num_links_" + column] = self.df[column]. \
                apply(lambda x: len(text_processor.TextProcessor(x).get_links()))
            text_features_df["has_hashtag_" + column] = self.df[column]. \
                apply(lambda x: len(text_processor.TextProcessor(x).get_hashtags()) > 0)
            has_cols.append("has_hashtag_" + column)
            self.df[column] = self.df[column]. \
                apply(lambda x: text_processor.TextProcessor(x).get_clean_text())
            text_features_df[column + "_len"] = self.df[column].replace(np.nan, '', regex=True).apply(len)
            text_features_df["num_words_" + column] = self.df[column].\
                apply(lambda x: text_processor.TextProcessor.count_words(x))
        self.add_features(text_features_df, True)
        self.convert_boolean(has_cols)

    def add_features(self, features, raw):
        """
        Add features to the df.
        :param features:
        :param raw: boolean says if the features are raw
        :return:
        """
        if raw or not self.raw:
            self.featured_columns += list(features.columns)
            self.df = pd.concat([self.df, features], axis=1)


    def generate_date_features(self, columns):

        date_features_df = pd.DataFrame()
        bool_cols = []
        for col in columns:
            date_features_df[col + "_year"] = self.df[col].dt.year
            date_features_df[col + "_month"] = self.df[col].dt.month
            date_features_df[col + "_is_year_end"] = self.df[col].dt.is_year_end
            bool_cols.append(col + "_is_year_end")
            date_features_df[col + "_is_year_start"] = self.df[col].dt.is_year_start
            bool_cols.append(col + "_is_year_start")
            date_features_df[col + "_quarter"] = self.df[col].dt.quarter
            date_features_df[col + "_is_quarter_end"] = self.df[col].dt.is_quarter_end
            bool_cols.append(col + "_is_quarter_end")
            date_features_df[col + "_is_quarter_start"] = self.df[col].dt.is_quarter_start
            bool_cols.append(col + "_is_quarter_start")

        self.add_features(date_features_df, True)
        self.convert_boolean(bool_cols)




    def generate_distance_from_created_at(self, columns, raw):

        date_features_df = pd.DataFrame()
        for col in columns:
            date_features_df["days_range_" + col] = (self.df[col]
                                                     - self.df["created_at"]).apply(lambda x: x.days if not pd.isnull(x) else -1)

        self.add_features(date_features_df, raw)


    def generate_has_features(self, columns):

        has_features_df = pd.DataFrame()
        for col in columns:
            has_features_df["has_" + col] = col in self.df and self.df[col] != np.nan

        self.add_features(has_features_df, True)
        self.convert_boolean(has_features_df.columns)


    @staticmethod
    def _get_length_list(x):
        try:
            return len(x)
        except:
            return 0

    def generate_count_features(self, columns):

        count_features_df = pd.DataFrame()
        for column in columns:
            count_features_df["num_" + column] = self.df[column].apply(DataPipeline._get_length_list)

        self.add_features(count_features_df, True)

    def convert_boolean(self, columns):
        for column in columns:
            self.df[column] = self.df[column].fillna(0) *1


    @staticmethod
    def is_relevant_country_US(location):
        if location is not None:
            return (location["country_code"] == "US") *1
        else:
            return 0

    @staticmethod
    def relevant_state(location):
        if location is not None:
            return location["state_code"]
        else:
            return None


    def feature_engineering(self):

        to_has = ["creator_photo",  "media", "photo", "video", "topic", "restricted_location"]
        to_count = ["languages",  "targets", "tags"]
        to_days_from_created_at = ["end_date", "last_past_verified_victory_date",
                        "last_past_victory_date"]
        to_days_from_created_at_not_raw = ["last_update"]
        to_date_features = ["created_at"]
        to_text_features = ["ask", "display_title", "description", "letter_body"]

        to_convert_boolean = ["comments_last_page", "is_en_US",
                              "is_organization", "is_pledge", "is_verified_victory"]

        self.generate_text_features(to_text_features)
        self.generate_date_features(to_date_features)
        self.generate_distance_from_created_at(to_days_from_created_at, True)
        self.generate_distance_from_created_at(to_days_from_created_at_not_raw, False)
        self.generate_has_features(to_has)
        self.generate_count_features(to_count)


        self.df["goal_days_ratio"] = self.df["calculated_goal"]/self.df["days_range_end_date"]

        self.df["is_en_US"] = self.df["original_locale"] == "en-US"

        self.convert_boolean(to_convert_boolean)

        # Get details from relevant location
        self.relevant_location_dummies()

        # Get dummies from topic
        self.topic_dummies()

        #Get Target information
        self.generate_target_features()

        #Get Languages
        self.df["is_en"] = self.df["languages"].apply(lambda x: 1 if x[0] == "en" else 0)

    def relevant_location_dummies(self):

        valid_states = ["HI", "AK", "FL", "SC", "GA", "AL", "NC", "TN",
                        "RI", "CT", "MA", "ME", "NH", "VT", "NY", "NJ",
                        "PA", "DE", "MD", "WV", "KY", "OH", "MI", "WY",
                        "MT", "ID", "WA", "DC", "TX", "CA", "AZ", "NV",
                        "UT", "CO", "NM", "OR", "ND", "SD", "NE", "IA",
                        "MS", "IN", "IL", "MN", "WI", "MO", "AR", "OK",
                        "KS", "LS", "VA"]
        dummies = pd.DataFrame()
        #Get details from relevant location
        dummies["relevant_country_US"] = self.df["relevant_location"].apply(lambda x: DataPipeline.is_relevant_country_US(x))
        self.df["relevant_state"] = self.df["relevant_location"].apply(lambda x: DataPipeline.relevant_state(x))
        for state in valid_states:
            dummies["relevant_state_" + state] = self.df["relevant_state"].apply(lambda x: 1 if x == state else 0)

        self.df["user_state"] = self.df["user"].apply(lambda x: x["state_code"])
        self.df["user_country"] = self.df["user"].apply(lambda x: x["country_code"])
        dummies["same_state"] = (self.df["relevant_state"] == self.df["user_state"])*1
        dummies["user_country_us"] = (self.df["user_country"] == "US")*1
        self.add_features(dummies, True)
    
    def topic_dummies(self):
        dummies = pd.DataFrame()
        valid_topics = ['environment', 'humanrights','criminaljustice',
           'economicjustice', 'animals', 'health', 'education',
           'gayrights', 'immigration', 'womensrights',
           'humantrafficking', 'food', 'socialentrepreneurship',
           'globalpoverty', 'race', 'homelessness']

        for topic in valid_topics:
            dummies["topic_" + topic] = self.df["topic"].apply(lambda x: 1 if x == topic else 0)
        self.add_features(dummies, True)


    def generate_target_features(self):
        target_features = pd.DataFrame()
        tp = TargetsProcessor()
        target_features["count_past_responses"] = self.df[["targets", "id"]]. \
            apply(lambda x: tp.get_count_past_responses(x[0],x[1]), axis= 1)
        target_features["count_democrat_targets"] = self.df["targets"]. \
            apply(lambda x: tp.get_count_democrat_targets(x))
        target_features["count_republican_targets"] = self.df["targets"]. \
            apply(lambda x: tp.get_count_republican_targets(x))
        target_features["count_not_found_targets"] = self.df["targets"]. \
            apply(lambda x: tp.get_count_not_found_target(x))
        target_features["count_custom_targets"] = self.df["targets"]. \
            apply(lambda x: tp.get_count_customs(x) )
        target_features["count_group_targets"] = self.df["targets"]. \
            apply(lambda x: tp.get_count_groups(x))

        self.featured_columns += list(target_features.columns)
        self.df = pd.concat([self.df, target_features], axis=1)


    def get_filtered_df(self, cache_data = False):
        self.df = self.df[self.featured_columns]

        if cache_data:
            conn = MongoConnection.default_connection()
            featured_petitions = conn['changeorg']['featured_petitions']

            records = json.loads(self.df.T.to_json()).values()
            featured_petitions.insert(records)

        return self.df

    def convert_target(self):
        self.df[self.target] = self.df[self.target].apply(lambda x: 1 if x == "victory" else 0)

    def outliers(self):
        # Trying to remove the trolled victories, should be done in more reliable way in future versions
        # I consider trolls the victories that have less than 100 signatures
        self.df = self.df[(self.df['status'] == 0) | ((self.df['status'] == 1) & (self.df['displayed_signature_count'] > 100))]

    def apply_pipeline(self):
        self.clean_data()
        self.feature_engineering()
        self.convert_target()
        if not self.to_predict:
            self.outliers()
        self.df.fillna(0, inplace=True)
        return self.get_filtered_df()


    def save_df(self):
        conn = MongoConnection.default_connection()
        collection = 'featured_petitions'
        if self.raw:
            collection = 'featured_petitions_raw'
        conn['changeorg'].drop_collection(collection)
        collection = conn['changeorg'][collection]
        records = json.loads(self.df.T.to_json()).values()
        collection.insert(records)


if __name__ == "__main__":
    conn = MongoConnection.default_connection()
    petitions_scraped = conn['changeorg']['petitions_scraped']

    cursor = petitions_scraped.find( {"id": {"$gt":0}})
    data = pd.DataFrame(list(cursor))

    data_pipeline = DataPipeline(data, False, False)
    f_data = data_pipeline.apply_pipeline()
    data_pipeline.save_df()

    print f_data.shape



