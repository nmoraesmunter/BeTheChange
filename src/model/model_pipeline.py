from __future__ import division
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, \
    roc_curve, auc, precision_recall_curve
from utils.utils import read_mongo
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import numpy as np
from time import time
from pprint import pprint
from utils.utils import save_model

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import RepeatedEditedNearestNeighbours

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN

from imblearn.ensemble import EasyEnsemble
from imblearn.ensemble import BalanceCascade



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
        # 'Random under-sampling'
        # CondensedNearestNeighbour(size_ngh=51, n_seeds_S=51)
        #Accuracy: 0.939693267481
        #Precision: 0.238095238095
        #Recall: 0.897435897436
        #SMOTE(ratio=ratio, kind='regular')
        #Accuracy: 0.962568234988
        #Precision: 0.324468085106
        #Recall: 0.782051282051
        #SMOTE(ratio=ratio, kind='borderline1')
        #Accuracy: 0.971146347803
        #Precision: 0.372093023256
        #Recall: 0.615384615385
        #SMOTE(ratio=ratio, kind='borderline2')
        #Accuracy: 0.965427605927
        #Precision: 0.333333333333
        #Recall: 0.705128205128
        #svm_args = {'class_weight': 'auto'}
        #svmsmote = SMOTE(ratio=ratio, kind='svm', **svm_args)
        #Accuracy: 0.972186119054
        #Precision: 0.395683453237
        #Recall: 0.705128205128

        ratio = 'auto'

        US = EasyEnsemble()

        X, y = US.fit_sample(X.toarray(), y)
        weights = np.array([1/y.mean() if i == 1 else 1 for i in y])
        return super(RandomForestClassifier, self).fit(X,y) #,sample_weight=weights)

class WeightedAdaClassifier(AdaBoostClassifier):
    def fit(self, X , y = None):
        weights = np.array([1/y.mean() if i == 1 else 1 for i in y])
        return super(AdaBoostClassifier, self).fit(X,y,sample_weight=weights)


class ModelPipeline(object):

    def __init__(self, clf):


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
                imp[feat] * 100, 3)) for feat in np.argsort(imp)[-1:-(n+1):-1])
        else:
            return self.columns[np.argsort(imp)[-1:-(n+1):-1]], \
                sorted(imp)[-1:-(n+1):-1]

    def grid_search(self, X, y):

        parameters = {
            'clf__n_estimators': [100, 200, 300] ,
            'clf__max_features': ['sqrt', 50, 80],
            'clf__max_depth': [None, 50, 100],
            'clf__oob_score': [False, True],
            'clf__random_state':[29],
            'clf__class_weight':['balanced', None, 'balanced_subsample'],
            'clf__min_samples_split': [2, 10, 20]
        }


        grid_search = GridSearchCV(self.pipeline, parameters, n_jobs=-1, verbose=1, scoring = "recall")

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
                    {"$and": [{"created_at_year": {"$gt": 2014}}
                       # , {"displayed_signature_count":  {"$gt": 99}}
                       # , {"status": 1 }
                              ]})


    df.fillna(0, inplace=True)
    df.pop("display_title")
    df.pop("letter_body")
    df.pop("id")
    df.pop("_id")
   # df.pop("displayed_signature_count")
   # df.pop("displayed_supporter_count")
    #df.pop("progress")
    #df.pop("is_verified_victory")
    # df.pop("description")
   # df["status"] = df["status"].apply(lambda x: 1 if x == "victory" else 0)
    print df.shape
    y = df.pop("status")
    df.pop("is_verified_victory")
   # y = df.pop("is_verified_victory")
    X = df


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    parameters = {
        'n_estimators': 300,
        'max_features': 80,
        'max_depth': None,
        'min_samples_leaf' : 20,
        'random_state':29,
        'class_weight': None
    }

    rfc = WeightedRFClassifier()
    rfc.set_params(**parameters)

    model_pipeline = ModelPipeline(rfc)
    #best_params = model_pipeline.grid_search(X_train, y_train)

    model_pipeline.fit(X_train, y_train)

    save_model(model_pipeline, "rf_model_petitions")

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

    print model_pipeline.feat_importances(10)



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
  #  plt.show()
'''
'''


