from preprocess import text_processor


def get_text_features(df, column):

    df["num_capitalized_words_" + column] = df[column].\
        apply(lambda x: text_processor.TextProcessor(x).count_capitalized_words())
    #df["num_words_" + column] = df[column].\
    #    apply(lambda x: text_processor.TextProcessor(x).count_words())
    df["num_bold_words_" + column] = df[column].\
        apply(lambda x: text_processor.TextProcessor(x).count_words_bold())
    df["num_italic_words_" + column] = df[column].\
        apply(lambda x: text_processor.TextProcessor(x).count_words_italic())
    #df["links_popularity_" + column] = df[column].\
    #    apply(lambda x: text_processor.TextProcessor(x).get_mean_link_popularity())
    df["num_links_" + column] = df[column].\
        apply(lambda x: len(text_processor.TextProcessor(x).get_links()))
    df["has_hashtag_" + column] = df[column].\
        apply(lambda x: len(text_processor.TextProcessor(x).get_hashtags()) > 0)

    return df
