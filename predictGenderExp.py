import sacred
ex = sacred.Experiment(name="gender_predict")


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, mean_squared_error, r2_score, accuracy_score
import util_funcs
import data_reader as read

@ex.named_config
def rf():
    parameters = {
        'rf__criterion': ["gini", "entropy"],
        'rf__n_estimators': [100, 200, 400, 600],
        'rf__max_features' : ['auto', 'log2', .1, .4, .8],
        'rf__max_depth' : [None, 2, 4],
        'rf__min_samples_split' : [2,8],
        'rf__n_jobs' : [-1],
        'rf__min_weight_fraction_leaf' : [0, 0.2, 0.5]
    }
    clf_name = "rf"
    clf_step = ('rf', RandomForestClassifier())

@ex.named_config
def lr():
    parameters = {
        'lr__tol': [0.001, 0.0001, 0.00001],
        'lr__multi_class': ["multinomial"],
        'lr__C': [0.05, .1, .2],
        'lr__solver': ["sag"],
        'lr__max_iter': [250],
        'lr__n_jobs': [-1]
    }
    clf_name = "lr"
    clf_step = ('lr', LogisticRegression())

@ex.named_config
def debug():
    num_files = 300

@ex.config
def config():
    parameters = {}
    clf_step = None
    clf_name = ''
    num_files = None


@ex.main
def main():
    print("hi")

if __name__ == "__main__":
    ex.run_commandline()
