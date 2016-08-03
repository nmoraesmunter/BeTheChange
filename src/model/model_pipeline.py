from __future__ import division
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, \
    roc_curve, auc, precision_recall_curve
from utils.utils import read_mongo
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from time import time
from utils.utils import save_model


class ColumnExtractor(BaseEstimator):

    def __init__(self, column_name):
        self.column_name = column_name

    def transform(self, X):
        return X[self.column_name].values

    def fit(self, X, y=None):
        return self


class ColumnPop(BaseEstimator):

    def __init__(self, column_name):
        self.column_name = column_name

    def transform(self, X):
        X = X.drop(self.column_name, axis=1)
        return X.values

    def fit(self, X, y=None):
        return self

class WeightedRFClassifier(RandomForestClassifier):
    def fit(self, X , y = None):
        weights = np.array([1/y.mean() if i == 1 else 1 for i in y])
        return super(RandomForestClassifier, self).fit(X,y,sample_weight=weights)

class WeightedAdaClassifier(AdaBoostClassifier):
    def fit(self, X , y = None):
        weights = np.array([1/y.mean() if i == 1 else 1 for i in y])
        return super(AdaBoostClassifier, self).fit(X,y,sample_weight=weights)


class ModelPipeline(object):

    def __init__(self, clf, params = {}):

        self.columns =[]
        self.count_vectorizer = CountVectorizer(stop_words="english", max_features=100)

        self.pipeline = Pipeline([
            ('features', FeatureUnion([
                ('nlp', Pipeline([
                    ('extract', ColumnExtractor("description")),
                    ('counts', self.count_vectorizer),
                    ('tf_idf', TfidfTransformer())
                ])),
                ('non-nlp', Pipeline([
                ('pop', ColumnPop("description"))
                ]))
            ])),
            ('clf', clf)
            ])
        self.pipeline.set_params(**params)

    def fit(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)
        nlp_col = ['tf_%s' % x for x in self.count_vectorizer.get_feature_names()]
        non_nlp_col = list(X_train.columns.drop("description"))
        self.columns = nlp_col + non_nlp_col

    def predict(self, X_test):
        return self.pipeline.predict(X_test)

    def feat_importances(self, n=20, string=True):

        imp = self.pipeline.steps[1][1].feature_importances_
        if string:
            return ''.join('%s: %s%%\n' % (self.columns[feat], round(
                imp[feat] * 100, 1)) for feat in np.argsort(imp)[-1:-(n+1):-1])
        else:
            return self.columns[np.argsort(imp)[-1:-(n+1):-1]], \
                sorted(imp)[-1:-(n+1):-1]

    def grid_search(self, X, y):

        rf_parameters = {
           'clf__n_estimators': [200],
            'clf__max_features': ['sqrt', 50, 80],
            'clf__max_depth' : [50, 100],
            'clf__oob_score': [False],
            'clf__min_samples_split':[20, 25]
        }

        ada_parameters = {
           'clf__n_estimators': [200],
            'clf__max_features': ['sqrt', 50, 80],
            'clf__max_depth' : [50, 100],
            'clf__oob_score': [False],
            'clf__min_samples_split':[20, 25]
        }

        param_grid = {"base_estimator__criterion": ["gini", "entropy"],
                      "base_estimator__splitter": ["best", "random"],
                      "n_estimators": [1, 2]
                      }

        grid_search = GridSearchCV(self.pipeline, parameters, n_jobs=-1, verbose=1)

        print("Performing grid search...")
        print("pipeline:", [name for name, _ in self.pipeline.steps])
        print("parameters:")
        pprint(parameters)
        t0 = time()
        grid_search.fit(X, y)
        print("done in %0.3fs" % (time() - t0))
        print()

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
        return best_parameters

if __name__ == "__main__":

    df = read_mongo("changeorg", "featured_petitions",
                    {"$and":[{"displayed_signature_count": {"$gt": 100}},
                             {"created_at_year": {"$gt": 2014}},
                            {"status": {"$in": ["victory", "closed"]}},
                             {"languages_en":1}]})

    df.pop("display_title")
    df.pop("letter_body")
    df.pop("id")
    df.pop("_id")
    #  df.pop("displayed_signature_count")
    #  df.pop("displayed_supporter_count")
    df.pop("is_verified_victory")
    # df.pop("description")
    df["status"] = df["status"].apply(lambda x: 1 if x == "victory" else 0)
    print df.shape
    y = df.pop("status")
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    parameters = {
        'n_estimators': 200,
        'max_features': 'sqrt',
        'max_depth': None
    }
    parameters = {
    }

    rf = WeightedRFClassifier()
    rf.set_params(**parameters)
    ada = WeightedAdaClassifier(**parameters)
 #   vc = VotingClassifier(estimators=[('rf', rf), ('ada', ada)], voting='hard')

    model_pipeline = ModelPipeline(ada)
  #  bp = model_pipeline.grid_search(X_train, y_train)

    parameters = {
        'clf__n_estimators': 200,
        'clf__max_features': 50,
        'clf__max_depth': 100,
        'clf__oob_score': False,
        'clf__min_samples_split': 25
    }

   # model_pipeline = ModelPipeline(rf)
    model_pipeline.fit(X_train, y_train)

    save_model(model_pipeline, "model")


    y_pred_train = model_pipeline.predict(X_train)
    y_pred = model_pipeline.predict(X_test)


    print "--------------------------TRAIN-----------------------------------"
    print "victories:" , sum(y_train)
    print "total:" , len(y_train)
    print "null accuracy:" , 1-(sum(y_train) / len(y_train))
    print "Accuracy:", accuracy_score(y_train, y_pred_train)
    print "Precision:", precision_score(y_train, y_pred_train)
    print "Recall:", recall_score(y_train, y_pred_train)

    print "--------------------------TEST-----------------------------------"
    print "victories:", sum(y_test)
    print "total:", len(y_test)
    print "null accuracy:", 1 - (sum(y_test) / len(y_test))
    print "Accuracy:", accuracy_score(y_test, y_pred)
    print "Precision:", precision_score(y_test, y_pred)
    print "Recall:", recall_score(y_test, y_pred)
    print "confusion matrix"
    print confusion_matrix(y_test, y_pred, [1, 0])


     #Print the feature ranking
    print("------------------Feature ranking--------------------------------------")

    print model_pipeline.feat_importances(20)

    y_score = model_pipeline.pipeline.predict_proba(X_test)[:,1]

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_score)

    roc_auc = auc(false_positive_rate, true_positive_rate)


    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()








    """
    Calculate accuracy, precision, recall for varying thresholds.

    thresholds = y_score
    thresholds.sort()
    thresh_acc = []
    thresh_prec = []
    thresh_rec = []
    for threshold in thresholds:
        y_predicted = []
        for ytp in y_score:
            y_predicted.append(int(ytp >= threshold))
        thresh_acc.append(accuracy_score(y_test, y_predicted))
        thresh_prec.append(precision_score(y_test, y_predicted))
        thresh_rec.append(recall_score(y_test, y_predicted))
    plt.plot(thresholds, thresh_acc, label='accuracy')
    plt.plot(thresholds, thresh_prec, label='precision')
    plt.plot(thresholds, thresh_rec, label='recall')
    plt.legend()
    plt.show()
    """