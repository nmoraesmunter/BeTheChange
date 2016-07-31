from __future__ import division
import pandas as pd
from pymongo import MongoClient
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from preprocess.data_pipeline 


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer as lemmatizer

from utils.utils import save_model

class Model(object):

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.columns = []
        self.tf_col = []



    def vectorize_text(self, df):

        self.vectorizer = TfidfVectorizer(stop_words='english', use_idf=False,
                                   max_features=40)
        vect = self.vectorizer.fit_transform(df["description"].values)
        self.tf_col = ['tf_%s' % x for x in self.vectorizer.get_feature_names()]
        tf_df = pd.DataFrame(vect.toarray(), columns=self.tf_col)
        df = df.drop("description", axis=1)
        df = df.reset_index()
        df = df.drop("index", axis=1)
        df = pd.concat([df, tf_df], axis=1)
        return df





    def filter_features(self, df):
        df = df[["status", "num_past_petitions", "num_past_verified_victories", "num_past_victories",
                 "num_comments", "title_len", "overview_len", "news_coverages",
                 "letter_body_len", "milestones", "ask_len", "display_title_len", "description_len",
                 "days_range_end_at", "calculated_goal", "num_targets", "comments_likes", "fb_popularity",
                 "goal", "creator_description_len", "creator_mission_len", "creator_type_user",
                 "num_tweets", "comments_likes", "endorsements", "signature_count", "num_capitalized_words_description",
                 "num_bold_words_description", "num_italic_words_description",
                 "num_links_description", "has_hashtag_description", "description",
                 "created_at_year", "created_at_month",
                 "created_at_is_year_end",
                 "created_at_is_year_start",
                 "created_at_quarter",
                 "created_at_is_quarter_end",
                 "created_at_is_quarter_start"]]
        return df

    def filter_features_new_petition(self, df):
        df = df[["status", "num_past_petitions", "num_past_verified_victories", "num_past_victories",
                 "title_len", "overview_len",
                 "letter_body_len", "ask_len", "display_title_len", "description_len",
                 "days_range_end_at", "calculated_goal", "num_targets",
                 "goal", "creator_description_len", "creator_mission_len",
                 "creator_type_user", "num_capitalized_words_description",
                 "num_bold_words_description", "num_italic_words_description",
                 "num_links_description", "has_hashtag_description", "description",
                 "created_at_year", "created_at_month",
                 "created_at_is_year_end",
                 "created_at_is_year_start",
                 "created_at_quarter",
                 "created_at_is_quarter_end",
                 "created_at_is_quarter_start"]]
        return df

    def filter_features_all(self, df):

        df = df[
           ['status','calculated_goal','comments_likes','creator_has_address'
            ,'creator_has_contact_email','creator_has_fb_page','creator_has_slug'
            ,'creator_has_twitter','creator_has_verified_at'
            ,'creator_has_verified_by','creator_has_verified_req'
            ,'creator_has_video','creator_has_website','discoverable'
            ,'endorsements','fb_popularity','goal','has_video','is_pledge'
            ,'milestones','news_coverages','num_comments','num_past_petitions'
            ,'num_past_verified_victories','num_past_victories','num_tweets'
            ,'signature_count','ask_len','category_Criminal Justice'
            ,'category_Economic Justice','category_Education'
            ,'category_End Sex Trafficking','category_Environment'
            ,'category_Gay Rights','category_Global Poverty','category_Health'
            ,'category_Homelessness','category_Human Rights'
            ,'category_Human Trafficking','category_Immigrant Rights'
            ,'category_Race in America','category_Social Entrepreneurship'
            ,'category_Sustainable Food' , "category_Women's Rights"
            ,'creator_country_AF','creator_country_AQ','creator_country_AR'
            ,'creator_country_AT','creator_country_AU','creator_country_AZ'
            ,'creator_country_BA','creator_country_BD','creator_country_BE'
            ,'creator_country_BI','creator_country_BR','creator_country_CA'
            ,'creator_country_CH','creator_country_CI','creator_country_CL'
            ,'creator_country_CM','creator_country_CN','creator_country_CO'
            ,'creator_country_DE','creator_country_DO','creator_country_DZ'
            ,'creator_country_EC','creator_country_EG','creator_country_ES'
            ,'creator_country_FM','creator_country_FR','creator_country_GB'
            ,'creator_country_GE','creator_country_GH','creator_country_GI'
            ,'creator_country_GR','creator_country_GT','creator_country_GU'
            ,'creator_country_HR','creator_country_HU','creator_country_ID'
            ,'creator_country_IE','creator_country_IL','creator_country_IN'
            ,'creator_country_IQ','creator_country_IR','creator_country_IT'
            ,'creator_country_JM','creator_country_JO','creator_country_JP'
            ,'creator_country_KE','creator_country_KR','creator_country_KV'
            ,'creator_country_KY','creator_country_KZ','creator_country_LT'
            ,'creator_country_MA','creator_country_MM','creator_country_MX'
            ,'creator_country_MY','creator_country_MZ','creator_country_NG'
            ,'creator_country_NI','creator_country_NL','creator_country_NO'
            ,'creator_country_NP','creator_country_NZ','creator_country_PA'
            ,'creator_country_PH','creator_country_PK','creator_country_PL'
            ,'creator_country_PR','creator_country_PS','creator_country_PT'
            ,'creator_country_RO','creator_country_RS','creator_country_RU'
            ,'creator_country_RW','creator_country_SA','creator_country_SD'
            ,'creator_country_SE','creator_country_SG','creator_country_SN'
            ,'creator_country_SV','creator_country_TH','creator_country_TN'
            ,'creator_country_TR','creator_country_TT','creator_country_TW'
            ,'creator_country_TZ','creator_country_UA','creator_country_UG'
            ,'creator_country_UM','creator_country_US','creator_country_VE'
            ,'creator_country_VI','creator_country_VN','creator_country_ZA'
            ,'creator_country_ZW','creator_description_len','creator_locale_de'
            ,'creator_locale_en-AU','creator_locale_en-CA','creator_locale_en-GB'
            ,'creator_locale_en-IN','creator_locale_en-US','creator_locale_es'
            ,'creator_locale_es-419','creator_locale_es-AR','creator_locale_fr'
            ,'creator_locale_fr-FR','creator_locale_id','creator_locale_it'
            ,'creator_locale_ja','creator_locale_pt-BR','creator_locale_ru'
            ,'creator_locale_ru-RU','creator_locale_th','creator_locale_tr'
            ,'creator_mission_len','creator_state_0','creator_state_13'
            ,'creator_state_42','creator_state_A','creator_state_AA'
            ,'creator_state_AB','creator_state_AE','creator_state_AK'
            ,'creator_state_AL','creator_state_AN','creator_state_AP'
            ,'creator_state_AR','creator_state_AZ','creator_state_BC'
            ,'creator_state_BE','creator_state_BG','creator_state_CA'
            ,'creator_state_CO','creator_state_CT','creator_state_DC'
            ,'creator_state_DE','creator_state_DIF','creator_state_DL'
            ,'creator_state_ENG','creator_state_FL','creator_state_FM'
            ,'creator_state_GA','creator_state_GU','creator_state_HI'
            ,'creator_state_IA','creator_state_ID','creator_state_IL'
            ,'creator_state_IN','creator_state_KS','creator_state_KY'
            ,'creator_state_LA','creator_state_MA','creator_state_MB'
            ,'creator_state_MD','creator_state_ME','creator_state_MG'
            ,'creator_state_MI','creator_state_MN','creator_state_MO'
            ,'creator_state_MP','creator_state_MS','creator_state_MT'
            ,'creator_state_NC','creator_state_ND','creator_state_NE'
            ,'creator_state_NH','creator_state_NJ','creator_state_NM'
            ,'creator_state_NSW','creator_state_NV','creator_state_NY'
            ,'creator_state_OH','creator_state_OK','creator_state_ON'
            ,'creator_state_OR','creator_state_PA','creator_state_PR'
            ,'creator_state_QC','creator_state_RI','creator_state_RP'
            ,'creator_state_SC','creator_state_SCT','creator_state_SD'
            ,'creator_state_TN','creator_state_TX','creator_state_TXQ'
            ,'creator_state_UT','creator_state_VA','creator_state_VI'
            ,'creator_state_VT','creator_state_WA','creator_state_WI'
            ,'creator_state_WLS','creator_state_WV','creator_state_WY'
            ,'creator_type_user','description_len','display_title_len'
            ,'days_range_end_at','num_languages','letter_body_len'
            ,'original_locale_en-AU','original_locale_en-CA','original_locale_en-GB'
            ,'original_locale_en-IN','original_locale_en-US','original_locale_es'
            ,'original_locale_es-419','original_locale_es-AR','original_locale_fr'
            ,'original_locale_id','original_locale_it','original_locale_ja'
            ,'original_locale_pt-BR','original_locale_th','original_locale_tr'
            ,'overview_len','num_targets','title_len', 'num_capitalized_words_description'
            ,'num_bold_words_description', 'num_italic_words_description'
            ,'num_links_description', 'has_hashtag_description', 'description']]
        return df

    def remove_data_leakage(self, df):
        df["num_past_petitions"] = df["num_past_petitions"].apply(lambda x: x - 1 if x >0 else x)
        df["num_past_verified_victories"] =  df["num_past_verified_victories"].apply(lambda x: x - 1 if x >0 else x)
        df["num_past_victories"] = df["num_past_victories"].apply(lambda x: x - 1 if x >0 else x)
        return df

    def fit_weighted_rf(self, X, y,  weight=1, leaf=15, trees=40):
        self.model = RandomForestClassifier(
            n_estimators=trees,
            min_samples_leaf=leaf, max_features='sqrt', max_depth=None)
        weights = np.array([weight/(y.mean()) if x else 1 for x in list(y)])
        self.model.fit(X, y, sample_weight=weights)


    def data_pipeline(self, df):
        df = data_engineering.get_text_features(df, "description")
        df = clean_data(df)
        df = self.remove_data_leakage(df)
        df = self.filter_features_new_petition(df)
        return df

    def fit(self, X_train, y_train):
        X_train = self.vectorize_text(X_train)
        self.columns = X_train.columns
        self.fit_weighted_rf(X_train.values, y_train)

    def predict(self, X_test):

        vect = self.vectorizer.transform(X_test["description"])
        tf_df = pd.DataFrame(vect.toarray(), columns=self.tf_col)
        X_test = X_test.reset_index()
        X_test = X_test.drop("index", axis=1)
        X_test = pd.concat([X_test, tf_df], axis=1)
        new_cols = set(self.columns).difference(set(X_test.columns))
        del_cols = set(X_test.columns).difference(set(self.columns))
        X_test = X_test.drop(list(del_cols), axis=1)
        for new_col in new_cols:
            X_test[new_col] = 0
        return self.model.predict(X_test.values)

    def feat_imp(self, n=20, string=True):

        imp = self.model.feature_importances_
        if string:
            return ''.join('%s: %s%%\n' % (self.columns[feat], round(
                imp[feat] * 100, 1)) for feat in np.argsort(imp)[-1:-(n+1):-1])
        else:
            return self.columns[np.argsort(imp)[-1:-(n+1):-1]], \
                sorted(imp)[-1:-(n+1):-1]


if __name__ == "__main__":
    valid_states = ["HI", "AK", "FL", "SC", "GA", "AL", "NC", "TN",
                    "RI", "CT", "MA", "ME", "NH", "VT", "NY", "NJ",
                    "PA", "DE", "MD", "WV", "KY", "OH", "MI", "WY",
                    "MT", "ID", "WA", "DC", "TX", "CA", "AZ", "NV",
                    "UT", "CO", "NM", "OR", "ND", "SD", "NE", "IA",
                    "MS", "IN", "IL", "MN", "WI", "MO", "AR", "OK",
                    "KS", "LS", "VA"]


    petitions_model = Model()

    data = read_mongo("changeorg", "us_closed_petitions",
                      {"$and": [{"creator_state": {"$in":valid_states}},
                                {"signature_count" : {"$gt": 100}},
                                {"endorsements": { "$exists": True }}]})

    data_df = petitions_model.data_pipeline(data)

    y = data_df.pop("status")
    X = data_df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    petitions_model.fit(X_train, y_train)

    save_model(petitions_model, "rf_new_petitions_model")

    y_pred_train = petitions_model.predict(X)
    y_pred = petitions_model.predict(X_test)


    print "--------------------------TRAIN-----------------------------------"
    print "victories:" , sum(y)
    print "total:" , len(y)
    print "null accuracy:" , 1-(sum(y) / len(y))
    print "Accuracy:", accuracy_score(y, y_pred_train)
    print "Precision:", precision_score(y, y_pred_train)
    print "Recall:", recall_score(y, y_pred_train)


    print "--------------------------TEST-----------------------------------"
    print "victories:", sum(y_test)
    print "total:", len(y_test)
    print "null accuracy:", 1-(sum(y_test) / len(y_test))
    print "Accuracy:", accuracy_score(y_test, y_pred)
    print "Precision:", precision_score(y_test, y_pred)
    print "Recall:", recall_score(y_test, y_pred)
    print "confusion matrix"
    print confusion_matrix(y_test, y_pred, [1, 0])

    importances = petitions_model.model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    print petitions_model.feat_imp(50)
