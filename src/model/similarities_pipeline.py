import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.cross_validation import train_test_split
from src.utils.utils import read_mongo
from src.utils.utils import save_model


class SimilaritiesPipeline(object):

    def __init__(self):
        self.pipeline = Pipeline([
                    ('counts', CountVectorizer(stop_words="english")),
                    ('tf_idf', TfidfTransformer())])

    def fit_transform(self, X):
        self.petition_ids = X["id"].reset_index(drop= True)
        self.tf_matrix = self.pipeline.fit_transform(X["description"])
        return self.tf_matrix

    def top_similar_petitions(self, new_description, n=4):
        tokenized_descr = self.pipeline.transform([new_description])
        cosine_similarities = linear_kernel(tokenized_descr, self.tf_matrix)
        return SimilaritiesPipeline.get_top_values(cosine_similarities[0], n, self.petition_ids)

    @staticmethod
    def get_top_values(similiarities, n, petition_ids):
        return [petition_ids[i] for i in np.argsort(similiarities)[-1:-n - 1:-1]]


if __name__ == "__main__":
    df = read_mongo("changeorg", "featured_petitions",
                    {"$and": [{"created_at_year": {"$gt": 2014}}
                              ]})

    y = df.pop("status")
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipeline = SimilaritiesPipeline()

    model_pipeline.fit_transform(X_train)

    save_model(model_pipeline, "similarities_model")

    print model_pipeline.top_similar_petitions("animals")
