import pandas as pd
import numpy as np
from catboost import CatBoostClassifier as CXGB
from sklearn import preprocessing as asd


from utils import to_file

if __name__ == '__main__':
    balance_data = pd.read_csv(
        'input/X_y_train.csv')

    max_abs_scaler = asd.MaxAbsScaler()

    X = balance_data.values[0:, :-1]
    Y = balance_data.values[0:, 1232]
    output = pd.read_csv('input/X_test.csv')
    X_test = output.values[0:, 1:]

    dtc = CXGB()

    max_abs_scaler.fit(np.concatenate((X, X_test)))
    X_train_maxabs = max_abs_scaler.transform(X)
    dtc.fit(X_train_maxabs, Y)
    X_test = max_abs_scaler.transform(X_test)
    predictions = dtc.predict(X_test)
    to_file(predictions, "y_test_submission_example")
