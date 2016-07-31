from __future__ import division
import pandas as pd
import numpy as np
from utils.utils import read_mongo
import text_processor


class DataPipeline(object):

    def __init__(self, df):
        self.df = df
        self.featured_columns = ["calculated_goal", "comments_likes", "displayed_signature_count",
                                 "displayed_supporter_count", "endorsements", "is_pledge", "is_organization",
                                 "fb_popularity", "links_fb_popularity", "milestones", "news_coverages",
                                 "num_comments", "num_past_petitions", "num_past_verified_victories",
                                 "num_past_victories", "num_responses", "num_tweets", "progress",
                                 "status", "twitter_popularity", "tweets_followers"]


    def drop_columns(self, to_drop):
        self.df.drop(to_drop, axis=1, inplace=True)

    def to_datetime(self, to_date_time):
        for col in to_date_time:
            self.df[col] = pd.to_datetime(self.df[col])

    def fill_nan_with_zeros(self, to_fill_with_zeros):
        for col in to_fill_with_zeros:
            self.df[col].fillna(0, inplace=True)

    def clean_data(self):

        to_drop = ["goal", "calculated_goal_with_endorsers", "creator_name", "title", "document_id",
                   "endorser_count", "organization", "petition_title", "photo_id", "primary_target",
                   "targeting_description", "total_signature_count", "total_supporter_count",
                   "victory_date", "victory_description", "weekly_signature_count"]
        to_date_time = ["created_at", "end_date", "last_past_verified_victory_date",
                        "last_past_victory_date", "last_update"]
        to_fill_with_zeros = ["displayed_signature_count"]

        self.drop_columns(to_drop)
        self.to_datetime(to_date_time)
        self.fill_nan_with_zeros(to_fill_with_zeros)

    def generate_text_features(self, columns):

        text_features_df = pd.DataFrame()
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
            text_features_df[column] = self.df[column]. \
                apply(lambda x: text_processor.TextProcessor(x).get_clean_text())
            text_features_df[column + "_len"] = self.df[column].replace(np.nan, '', regex=True).apply(len)
            text_features_df["num_words_" + column] = self.df[column].\
                apply(lambda x: text_processor.TextProcessor.count_words(x))

        self.featured_columns += list(text_features_df.columns)
        self.df = pd.concat([self.df, text_features_df], axis=1)



    def generate_date_features(self, columns):

        date_features_df = pd.DataFrame()
        for col in columns:
            date_features_df[col + "_year"] = self.df[col].dt.year
            date_features_df[col + "_month"] = self.df[col].dt.month
            date_features_df[col + "_is_year_end"] = self.df[col].dt.is_year_end
            date_features_df[col + "_is_year_start"] = self.df[col].dt.is_year_start
            date_features_df[col + "_quarter"] = self.df[col].dt.quarter
            date_features_df[col + "_is_quarter_end"] = self.df[col].dt.is_quarter_end
            date_features_df[col + "_is_quarter_start"] = self.df[col].dt.is_quarter_start

        self.featured_columns+= list(date_features_df.columns)
        self.df = pd.concat([self.df, date_features_df], axis=1)



    def generate_distance_from_created_at(self, columns):

        date_features_df = pd.DataFrame()
        for col in columns:
            date_features_df["days_range_" + col] = (self.df[col]
                                                     - self.df["created_at"]).apply(lambda x: x.days if not pd.isnull(x) else -1)

        self.featured_columns += list(date_features_df.columns)
        self.df = pd.concat([self.df, date_features_df], axis=1)


    def generate_has_features(self, columns):

        has_features_df = pd.DataFrame()
        for col in columns:
            has_features_df["has_" + col] = self.df[col] != np.nan

        self.featured_columns += list(has_features_df.columns)
        self.df = pd.concat([self.df, has_features_df], axis=1)
        self.convert_boolean(has_features_df.columns, False)


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

        self.featured_columns+= list(count_features_df.columns)
        self.df = pd.concat([self.df, count_features_df], axis=1)

    def convert_boolean(self, columns, add_to_featured_columns):
        for column in columns:
            self.df[column] = self.df[column].fillna(0) *1
        if add_to_featured_columns:
            self.featured_columns+=columns


    def feature_engineering(self):
        nlp = ["display_title", "description", "letter_body"]
        feature_engineering = ["creator_photo", "languages", "media", "photo", "video","original_locale", "targets",
                               "relevant_location", "restricted_location", "tags", "topic", "responses**", "user"]

        to_has = ["creator_photo",  "media", "photo", "video", "topic"]
        to_count = ["languages",  "targets", "tags"]
        to_days_from_created_at = ["end_date", "last_past_verified_victory_date",
                        "last_past_victory_date", "last_update"]
        to_date_features = ["created_at"]
        to_text_features = ["ask", "display_title", "description", "letter_body"]

        to_convert_boolean = ["comments_last_page","is_victory", "is_verified_victory", "is_en_US"]

        self.generate_text_features(to_text_features)
        self.generate_date_features(to_date_features)
        self.generate_distance_from_created_at(to_days_from_created_at)
        self.generate_has_features(to_has)
        self.generate_count_features(to_count)

        self.df["is_en_US"] = self.df["original_locale"] == "en-US"

        self.convert_boolean(to_convert_boolean, True)


    def get_filtered_df(self):
        return self.df[self.featured_columns]

    def apply_pipeline(self):
        data_pipeline.clean_data()
        data_pipeline.feature_engineering()
        return data_pipeline.get_filtered_df()

if __name__ == "__main__":

    data = read_mongo("changeorg", "petitions_scraped", {"id": {"$gt": 5000000}})

    data_pipeline = DataPipeline(data)
    data_pipeline.clean_data()
    data_pipeline.feature_engineering()
    f_data = data_pipeline.get_filtered_df()

    print f_data.shape



