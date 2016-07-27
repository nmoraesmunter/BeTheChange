from __future__ import division
import pandas as pd
from pymongo import MongoClient
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier



def read_mongo(database_name, collection_name, query={}, no_id=False):
    """ Read from Mongo and Store into DataFrame """

    # Get MongoDB
    mc = MongoClient()
    db = mc[database_name]
    petitions_col = db[collection_name]

    # Make a query to the specific DB and Collection
    cursor = db[collection_name].find(query)

    # Expand the cursor and construct the DataFrame
    df = pd.DataFrame(list(cursor))

    # Delete the _id
    if no_id:
        del df['_id']

    return df


def generate_length_column(df, column):
    df[column + "_len"] = df[column].replace(np.nan,'', regex=True).apply(len)


def _get_length_list(x):
    try:
        return len(x)
    except:
        return 0

def generate_num_column(df, column):
    df["num_" + column] = df[column].apply(_get_length_list)



def clean_data(df):

    nans_to_neg_1 = ['creator_zipcode']
    to_datetime = ['created_at',  'end_at', 'last_past_verified_victory_date', 'last_past_victory_date']
    to_days_range= ["end_at", 'last_past_verified_victory_date', 'last_past_victory_date']
    to_boolean = ['has_video', 'creator_has_website', 'creator_has_twitter',
                  'creator_has_address', 'creator_has_contact_email', 'creator_has_fb_page',
                  'creator_has_slug', 'creator_has_verified_at', 'creator_has_verified_by',
                  'creator_has_verified_req', 'creator_has_video', 'has_video', 'is_pledge',
                  'discoverable'
                  ]
    to_category =  ['category', 'creator_country', 'creator_locale', 'creator_state',
                   'creator_type', 'original_locale']
    to_drop = ['_id', 'ask', 'creator_city', 'creator_description', 'creator_display_name',
               'creator_first_name', 'creator_last_name', 'creator_mission', 'creator_name',
               'creator_org_name', 'creator_photo', 'creator_postal_code', 'creator_tax_country_code',
               'creator_tax_state_code', 'creator_url', 'description', 'display_title',
               'displayed_signature_count', 'image_url', 'languages', 'letter_body', 'targets','organization_name', 'organization_url',
               'overview', 'petition_id', 'photo', 'progress', 'tags','targets_detailed',
               'title', 'url', 'victory_date', 'created_at', 'creator_zipcode', 'creator_fb_permissions',
               'is_verified_victory', 'is_victory', 'last_update',  'days_range_last_past_verified_victory_date',
               'days_range_last_past_victory_date']
    to_length = ['ask', 'creator_description', 'creator_mission', 'description', 'display_title',
                 'letter_body', 'overview', 'title']
    to_num = ['languages', 'targets']

    to_has = ['']

    target = ['status']

    df = df[df["creator_type"].notnull()]

    columns = df.columns
    for col in columns:
        if col in nans_to_neg_1:
            df[col] = df[col].fillna(-1)
        if col in to_datetime:
            df[col] = pd.to_datetime(df[col])
        if col in to_days_range:
            df["days_range_" + col] = (df[col] - df["created_at"]).apply(lambda x: x.days if not pd.isnull(x) else -1)
            df.pop(col)
        if col in to_category:
            df[col] = df[col].astype('category')
            dummies = pd.get_dummies(df[col], drop_first=True).rename(columns=lambda x: col + "_" + str(x))
            df = pd.concat([df, dummies], axis=1)
            df.pop(col)
        if col in to_boolean:
            d = {True: 1, False: 0, np.nan: 0}
            df[col] = df[col].map(d)
        if col in to_length:
            generate_length_column(df, col)
        if col in to_num:
            generate_num_column(df, col)
        if col in to_has:
            df["has_" + col] = df[col] != np.nan
        if col in target:
            d = {"victory": 1, "closed": 0}
            df[col] = df[col].map(d)

    df.drop(to_drop, axis=1, inplace=True)
    return df


def sampling(df, target, target_ratio):
    current_ratio = df[target].value_counts()[1]/len(df[target])
    # oversample the victory cases
    if current_ratio < target_ratio:
        n = np.round(target_ratio * len(df[target])) - df[target].value_counts()[1]  # additional samples needed to meet target
        samples = df[df[target] == 1].sample(n, replace=True, random_state=0)
        df = df.append(samples, ignore_index=True, verify_integrity=True)
    # undersample the victory cases
    if current_ratio > target_ratio:
        n = np.round(target_ratio * len(df[target]))    # total target cases to meet ratio
        samples = df[df[target] == 1].sample(n, replace=False, random_state=0)
        df = df[df[target] == 0].append(samples, ignore_index=True, verify_integrity=True)
    return df

def filter_features(df):
    df["num_past_petitions"] = df["num_past_petitions"] - 1
    df["num_past_verified_victories"] =  df["num_past_verified_victories"] - 1
    df["num_past_victories"] = df["num_past_victories"] - 1
    df = df[["status","num_past_petitions", "num_past_verified_victories" , "num_past_victories",
             "num_comments", "title_len", "overview_len", "news_coverages",
             "letter_body_len", "milestones", "ask_len", "display_title_len", "description_len",
             "days_range_end_at", "calculated_goal", "num_targets", "comments_likes", "fb_popularity",
             "goal", "creator_description_len", "creator_mission_len", "creator_type_user",
             "num_tweets", "comments_likes", "endorsements", "signature_count"]]
    return df

if __name__ == "__main__":

    data = read_mongo("changeorg", "us_closed_petitions", {"endorsements": { "$exists": True }})

    clean_df = filter_features(clean_data(data))
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

   # model = AdaBoostClassifier(n_estimators=50)
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

