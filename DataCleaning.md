Data columns (total 66 columns):
_id                                565 non-null object
ask                                565 non-null object
calculated_goal                    565 non-null int64
calculated_goal_with_endorsers     565 non-null int64
comments_last_page                 565 non-null bool
comments_likes                     565 non-null int64
created_at                         565 non-null object
creator_name                       565 non-null object
creator_photo                      396 non-null object
description                        565 non-null object
discoverable                       565 non-null bool
display_title                      565 non-null object
displayed_signature_count          565 non-null int64
displayed_supporter_count          565 non-null int64
document_id                        565 non-null object
end_date                           565 non-null object
endorsements                       565 non-null int64
endorser_count                     565 non-null int64
fb_popularity                      565 non-null int64
goal                               0 non-null object
id                                 565 non-null int64
is_organization                    565 non-null bool
is_pledge                          565 non-null bool
is_verified_victory                565 non-null bool
is_victory                         565 non-null bool
languages                          565 non-null object
last_past_verified_victory_date    7 non-null object
last_past_victory_date             11 non-null object
last_update                        237 non-null object
letter_body                        565 non-null object
links_fb_popularity                565 non-null float64
media                              395 non-null object
milestones                         565 non-null int64
news_coverages                     565 non-null int64
num_comments                       565 non-null int64
num_past_petitions                 565 non-null int64
num_past_verified_victories        565 non-null int64
num_past_victories                 565 non-null int64
num_responses                      565 non-null int64
num_tweets                         565 non-null int64
organization                       92 non-null object
original_locale                    565 non-null object
petition_status                    565 non-null object
petition_title                     565 non-null object
photo                              395 non-null object
photo_id                           0 non-null float64
primary_target                     565 non-null object
progress                           565 non-null float64
relevant_location                  565 non-null object
restricted_location                37 non-null object
slug                               565 non-null object
status                             565 non-null object
tags                               565 non-null object
targeting_description              565 non-null object
targets                            565 non-null object
title                              424 non-null object
topic                              106 non-null object
total_signature_count              565 non-null int64
total_supporter_count              565 non-null int64
tweets_followers                   565 non-null int64
twitter_popularity                 565 non-null int64
user                               565 non-null object
victory_date                       95 non-null object
victory_description                188 non-null object
video                              16 non-null object
weekly_signature_count             565 non-null int64


to_drop = ["goal", "calculated_goal_with_endorsers", "creator_name", "title", "document_id", 
    "endorser_count", "organization", "petition_title", "photo_id", "primary_target",
     "targeting_description", "total_signature_count", "total_supporter_count", "user",
     "victory_date", "victory_description", ""weekly_signature_count]
     

to_keep = ["ask","calculated_goal", "comments_last_page", "comments_likes", 
    "creator_photo", "description", "display_title", "displayed_signature_count",
    "displayed_supporter_count", "endorsements", "is_victory", "is_verified_victory",
    "is_pledge", "is_organization", "id", "fb_popularity", "created_at", "end_date", "languages",
    "last_past_verified_victory_date", 
    "last_past_victory_date", "last_update", "letter_body", "links_fb_popularity", "media", "photo",
    "milestones", "news_coverages", "num_comments", "num_past_petitions", "num_past_verified_victories", 
    "num_past_victories", "num_responses num_tweets", "original_locale", "targets" ,"progress",
    "relevant_location", "restricted_location", "slug", "status", "tags", "topic"]
    
to_date_time = ["created_at", "end_date", "last_past_verified_victory_date", 
    "last_past_victory_date", "last_update"]
to_boolean = 
fill_na_0 = ["displayed_signature_count"]


nlp = ["ask", "description", "letter_body"]
feature_engineering["creator_photo", "languages", "media", "photo", "video" 
"original_locale", "targets", "relevant_location", "restricted_location",
"tags", "topic", "responses"]


