from __future__ import division
import pandas as pd
from pymongo import MongoClient
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score



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
                  'is_verified_victory', 'is_victory', 'discoverable'
                  ]
    to_category =  ['category', 'creator_country', 'creator_locale', 'creator_state',
                   'creator_type', 'original_locale']
    to_drop = ['_id', 'ask', 'creator_city', 'creator_description', 'creator_display_name',
               'creator_first_name', 'creator_last_name', 'creator_mission', 'creator_name',
               'creator_org_name', 'creator_photo', 'creator_postal_code', 'creator_tax_country_code',
               'creator_tax_state_code', 'creator_url', 'description', 'display_title',
               'displayed_signature_count', 'image_url', 'languages', 'letter_body', 'targets','organization_name', 'organization_url',
               'overview', 'petition_id', 'photo', 'progress', 'signature_count', 'tags','targets_detailed',
               'title', 'url', 'victory_date', 'created_at', 'creator_zipcode', 'creator_fb_permissions']
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



if __name__ == "__main__":

    data = read_mongo("changeorg", "us_closed_petitions", {"petition_id": {"$lt": 236494}})

    clean_df = clean_data(data)

    df_test = clean_df[:int(len(clean_df)*0.2)]
    df_train = clean_df[int(len(clean_df)*0.2):]
    df_train = sampling(df_train, 'status', 0.3)


    y = df_train.pop("status").values
    X = df_train

    y_test = df_test.pop("status").values
    X_test = df_test

    print "victories:" , sum(y_test)
    print "total:" , len(y_test)
    print "null accuracy:" , 1-(sum(y_test) / len(y_test))


    print "train victories:" , sum(y)
    print "train total:" , len(y)
    print "train null accuracy:" , 1-(sum(y) / len(y))

    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    print "Accuracy score:", accuracy_score(y, y_pred)


