import pandas as pd
import numpy as np
from catboost import CatBoostClassifier as CXGB
from sklearn import preprocessing as asd
import shap
import scikitplot as skplt
from sklearn.model_selection import train_test_split

from utils import to_file

if __name__ == '__main__':
    balance_data = pd.read_csv(
        'input/X_y_train.csv')

    max_abs_scaler = asd.MaxAbsScaler()

    X = balance_data.values[0:, :-1]
    Y = balance_data.values[0:, 1232]
    output = pd.read_csv('input/X_test.csv')
    X_test = output.values[0:, 1:]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.3)

    dtc = CXGB()

    max_abs_scaler.fit(np.concatenate((X, X_test)))
    X_train_maxabs = max_abs_scaler.transform(X)
    dtc.fit(X_train_maxabs, Y)
    X_test = max_abs_scaler.transform(X_test)
    predictions = dtc.predict(X_test)
    to_file(predictions, "y_test_submission_example")

    # SHAP
    explainer = shap.TreeExplainer(dtc)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)


    # AUC
    plt = skplt.metrics.plot_roc(Y_validation, predictions)
    plt.show()
