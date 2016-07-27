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
from src.preprocess.data_hygiene import filter_features_new_petition, filter_features, read_mongo, sampling, clean_data



if __name__ == "__main__":

    data = read_mongo("changeorg", "us_closed_petitions", {"endorsements": { "$exists": True }})

    clean_df = filter_features_new_petition(clean_data(data))
    np.random.seed(29)
    num_rows = clean_df.shape[0]
    test_sample_idx = np.random.choice(num_rows, round(num_rows * 0.3), replace=False)
    idx_df = clean_df.index.isin(test_sample_idx)
    df_test = clean_df.iloc[test_sample_idx]
    df_train = clean_df.iloc[~idx_df]
    df_train = sampling(df_train, 'status', 0.3)

    y = df_train.pop("status").values
    X = df_train

    y_test = df_test.pop("status").values
    X_test = df_test



    model =  RandomForestClassifier(n_estimators=20, criterion='gini',
                                    max_depth=None, min_samples_split=2,
                                    min_samples_leaf=1,
                                    min_weight_fraction_leaf=0.0,
                                    max_features='auto', max_leaf_nodes=None,
                                    bootstrap=True, oob_score=False, n_jobs=-1,
                                    verbose=0, warm_start=False,
                                    class_weight=None)

    #model = AdaBoostClassifier(n_estimators=50)
    #model  = MultinomialNB()
    model.fit(X, y)

    y_pred_train = model.predict(X)
    y_pred = model.predict(X_test)

    y_pred_proba = model.predict_proba(X_test)
    #  print y_pred_proba

    print "--------------------------TRAIN-----------------------------------"
    print "victories:" , sum(y)
    print "total:" , len(y)
    print "null accuracy:" , 1-(sum(y) / len(y))
    print "Accuracy:", accuracy_score(y, y_pred_train)
    print "Precision:", precision_score(y, y_pred_train)
    print "Recall:", recall_score(y, y_pred_train)


    print "--------------------------TEST-----------------------------------"
    print "victories:" , sum(y_test)
    print "total:" , len(y_test)
    print "null accuracy:" , 1-(sum(y_test) / len(y_test))
    print "Accuracy:", accuracy_score(y_test, y_pred)
    print "Precision:", precision_score(y_test, y_pred)
    print "Recall:", recall_score(y_test, y_pred)
    print "confusion matrix"
    print confusion_matrix(y_test, y_pred, [1, 0])

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(10):
        print("%d. feature %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))
