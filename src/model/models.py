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
from preprocess.data_hygiene import  read_mongo, clean_data
from preprocess import data_engineering

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
    df = df[["status","num_past_petitions", "num_past_verified_victories" , "num_past_victories",
             "num_comments", "title_len", "overview_len", "news_coverages",
             "letter_body_len", "milestones", "ask_len", "display_title_len", "description_len",
             "days_range_end_at", "calculated_goal", "num_targets", "comments_likes", "fb_popularity",
             "goal", "creator_description_len", "creator_mission_len", "creator_type_user",
             "num_tweets", "comments_likes", "endorsements", "signature_count", "num_capitalized_words_description",
             "num_bold_words_description", "num_italic_words_description",
            "num_links_description", "has_hashtag_description"]]
    return df


def filter_features_new_petition(df):
    df = df[["status","num_past_petitions", "num_past_verified_victories" , "num_past_victories",
            "title_len", "overview_len",
             "letter_body_len", "ask_len", "display_title_len", "description_len",
             "days_range_end_at", "calculated_goal", "num_targets",
             "goal", "creator_description_len", "creator_mission_len",
             "creator_type_user", "num_capitalized_words_description",
             "num_bold_words_description", "num_italic_words_description",
            "num_links_description", "has_hashtag_description"]]
    return df

def filter_features_all(df):

    df = df[
       ['calculated_goal','comments_likes','creator_has_address'
        ,'creator_has_contact_email','creator_has_fb_page','creator_has_slug'
        ,'creator_has_twitter','creator_has_verified_at'
        ,'creator_has_verified_by','creator_has_verified_req'
        ,'creator_has_video','creator_has_website','discoverable'
        ,'endorsements','fb_popularity','goal','has_video','is_pledge'
        ,'milestones','news_coverages','num_comments','num_past_petitions'
        ,'num_past_verified_victories','num_past_victories','num_tweets'
        ,'signature_count','status','ask_len','category_Criminal Justice'
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
        ,'num_links_description', 'has_hashtag_description']]
    return df


def remove_data_leakage(df):
    df["num_past_petitions"] = df["num_past_petitions"].apply(lambda x: x - 1 if x >0 else x)
    df["num_past_verified_victories"] =  df["num_past_verified_victories"].apply(lambda x: x - 1 if x >0 else x)
    df["num_past_victories"] = df["num_past_victories"].apply(lambda x: x - 1 if x >0 else x)
    return df

if __name__ == "__main__":

    data = read_mongo("changeorg", "us_closed_petitions", {"endorsements": { "$exists": True }})

    clean_df = clean_data(data)

    clean_df = data_engineering.get_text_features(clean_df, "description")

    clean_df = filter_features_new_petition(clean_df)

    clean_df = remove_data_leakage(clean_df)

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

    for f in range(20):
        print("%d. feature %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))

    dropped_columns = ['description',]
    columns = [
       'calculated_goal','comments_likes','creator_has_address'
        ,'creator_has_contact_email','creator_has_fb_page','creator_has_slug'
        ,'creator_has_twitter','creator_has_verified_at'
        ,'creator_has_verified_by','creator_has_verified_req'
        ,'creator_has_video','creator_has_website','discoverable'
        ,'endorsements','fb_popularity','goal','has_video','is_pledge'
        ,'milestones','news_coverages','num_comments','num_past_petitions'
        ,'num_past_verified_victories','num_past_victories','num_tweets'
        ,'signature_count','status','ask_len','category_Criminal Justice'
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
        ,'num_bold_words_description", "num_italic_words_description'
        ,'num_links_description', 'has_hashtag_description']