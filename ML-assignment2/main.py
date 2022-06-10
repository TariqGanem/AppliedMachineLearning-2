import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from utils import to_file

if __name__ == '__main__':
    balance_data = pd.read_csv(
        'input/X_y_train.csv')

    X = balance_data.values[0:, ]
    Y = balance_data.values[0:, 1232]
    dtc = RandomForestClassifier()
    dtc.fit(X, Y)
    output = pd.read_csv('input/X_test.csv')
    X_test = output.values[0:, ]
    predictions = dtc.predict(X_test)
    to_file(predictions, "y_test_submission_example", to_kaggle=False, msg="")
