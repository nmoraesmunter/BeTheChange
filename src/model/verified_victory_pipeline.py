from __future__ import division
import numpy as np
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
from time import time
from pprint import pprint
from utils.utils import save_model
from sklearn.linear_model import LogisticRegression

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





class WeightedRFClassifier(RandomForestClassifier):
    def fit(self, X , y = None):
        # 'Random under-sampling'
        # CondensedNearestNeighbour(size_ngh=51, n_seeds_S=51)
        #Accuracy: 0.939693267481
        #Precision: 0.238095238095
        #Recall: 0.897435897436

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

        smote = SMOTE(ratio='auto', kind='regular')
        X, y = smote.fit_sample(X, y)
       # weights = np.array([1/y.mean() if i == 1 else 1 for i in y])
        return super(RandomForestClassifier, self).fit(X,y)#,sample_weight=weights)

class WeightedAdaClassifier(AdaBoostClassifier):
    def fit(self, X , y = None):
        smote = SMOTE(ratio='auto', kind='regular')
        X, y = smote.fit_sample(X, y)
      #  weights = np.array([1/y.mean() if i == 1 else 1 for i in y])
        return super(AdaBoostClassifier, self).fit(X,y) #,sample_weight=weights)


class ModelPipeline(object):

    def __init__(self, clf):

        self.columns =[]

        self.pipeline = Pipeline([
            ('clf', clf)
            ])



    def fit(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)
        self.columns = list(X_train.columns)

    def predict(self, X_test):
        return self.pipeline.predict(X_test)


    def feat_importances(self, n=10, string=True):

        imp = self.pipeline.steps[0][1].feature_importances_
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


def generate_model( rf, model_name):

    collection = "featured_petitions"
    query = {"$and": [{"status": 1}]}
    target = "is_verified_victory"
  #  to_pop = "status"

    extract_features = ["goal_days_ratio", "milestones", "num_words_letter_body", "comments_likes",
                        "progress", "news_coverages", "created_at_quarter", "display_title_len",
                        "num_bold_words_description", "num_targets", "count_group_targets", "num_targets",
                        "num_capitalized_words_description", "num_capitalized_words_display_title",
                         "count_custom_targets", "count_democrat_targets", "count_republican_targets",
                        "is_organization", "is_verified_victory", "num_responses", "same_state"]

    df = read_mongo("changeorg", collection, query)
    df = df[df["days_range_end_date"] > 0]
    df = df[extract_features]
    df.fillna(0, inplace=True)
   # df.pop("display_title")
   # df.pop("letter_body")
   # df.pop("description")
   # df.pop("id")
   # df.pop("_id")
  #  df.pop(to_pop)
    y = df.pop(target)
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    ada_parameters = {
        'n_estimators': 300
    }

    rf_parameters = {
        'n_estimators': 300,
      #  'max_features': 80,
        'max_depth': None,
        'min_samples_leaf': 20,
        'random_state': 29,
        'class_weight': None
    }
    if rf:
        clf = WeightedRFClassifier()
        clf.set_params(**rf_parameters)
    else:
        clf = WeightedAdaClassifier()
        clf.set_params(**ada_parameters)

    model_pipeline = ModelPipeline(clf)

    model_pipeline.fit(X_train, y_train)

    save_model(model_pipeline, model_name)
    y_pred_train = model_pipeline.predict(X_train)
    y_pred = model_pipeline.predict(X_test)
    print "-------------------------------------------------------------------"
    print model_name
    print "--------------------------TRAIN-----------------------------------"
    print "victories:", sum(y_train)
    print "total:", len(y_train)
    print "null accuracy:", 1 - (sum(y_train) / len(y_train))
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

    # Print the feature ranking
    print("------------------Feature ranking--------------------------------------")

    print model_pipeline.feat_importances(100)

    y_score = model_pipeline.pipeline.predict_proba(X_test)[:,1]

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_score)

    roc_auc = auc(false_positive_rate, true_positive_rate)



if __name__ == "__main__":

    generate_model(True, "rf_verified_victories")
    generate_model(False, "ada_verified_victories")








