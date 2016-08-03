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
from preprocess.data_pipeline import DataPipeline
from sklearn.pipeline import  Pipeline, FeatureUnion
from utils.utils import read_mongo
from sklearn.feature_selection import SelectKBest


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



    def vectorize_text(self, train_df):

        self.vectorizer = TfidfVectorizer(stop_words='english', use_idf=False,
                                   max_features=40)
        vect = self.vectorizer.fit_transform(df["description"].values)
        self.tf_col = ['tf_%s' % x for x in self.vectorizer.get_feature_names()]
        tf_df = pd.DataFrame(vect.toarray(), columns=self.tf_col)
        train_df.drop("description", axis=1, inplace = True)
        #train_df = train_df.reset_index()
        #train_df.drop("index", axis=1, inplace = True)
        train_df = pd.concat([train_df, tf_df], axis=1)
        return train_df



    def fit_weighted_rf(self, X, y,  weight=1, leaf=15, trees=40):
        self.columns = X.columns
        self.model = RandomForestClassifier(
            n_estimators=trees,
            min_samples_leaf=leaf, max_features='sqrt', max_depth=None)
        weights = np.array([weight/(y.mean()) if x else 1 for x in list(y)])
        self.model.fit(X, y, sample_weight=weights)



    def fit(self, X_train, y_train):
        X_train = self.vectorize_text(X_train)
        self.columns = X_train.columns
        self.fit_weighted_rf(X_train.values, y_train)

    def predict(self, X_test):

#        vect = self.vectorizer.transform(X_test["description"])
#        tf_df = pd.DataFrame(vect.toarray(), columns=self.tf_col)
       # X_test = X_test.reset_index()
       # X_test = X_test.drop("index", axis=1)
 #       X_test = pd.concat([X_test, tf_df], axis=1)
 #       new_cols = set(self.columns).difference(set(X_test.columns))
 #       del_cols = set(X_test.columns).difference(set(self.columns))
 #       X_test = X_test.drop(list(del_cols), axis=1)
 #       for new_col in new_cols:
 #           X_test[new_col] = 0
        return self.model.predict(X_test.values)

    def feat_importances(self, n=20, string=True):

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
    nlp = ["display_title", "description", "letter_body"]

    df = read_mongo("changeorg", "featured_petitions",
                    {"$and":[{"displayed_signature_count": {"$gt": 100}},
                             {"created_at_year": {"$gt": 2010}},
                            {"status": {"$in": ["victory", "closed"]}},
                             {"languages_en":1}]})



    #data_pipeline = DataPipeline(data)

    #df = data_pipeline.apply_pipeline()



    petitions_model = Model()
    df.pop("display_title")
    df.pop("letter_body")
    df.pop("id")
    df.pop("_id")
  #  df.pop("displayed_signature_count")
  #  df.pop("displayed_supporter_count")
    df.pop("is_verified_victory")
    df.pop("description")
    df["status"] = df["status"].apply(lambda x: 1 if x == "victory" else 0)
    print df.shape
    y = df.pop("status")
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    petitions_model.fit_weighted_rf(X_train, y_train)

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

    print petitions_model.feat_importances(50)
