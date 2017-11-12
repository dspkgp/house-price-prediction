import quandl
import pandas as pd
import numpy as np
from sklearn import svm, preprocessing, cross_validation

style.use('fivethirtyeight')
api_key = "PS7aqn6uob9qjaFadDtE"  #froom Quandl account setting

def create_labels(cur_hpi, fut_hpi):
    if fut_hpi > cur_hpi:
        return 1
    else:
        return 0

housing_data = joblib.load('HPI.pkl')
housing_data = housing_data.pct_change()
housing_data.replace([np.inf, -np.inf], np.nan, inplace=True)
housing_data['US_HPI_future'] = housing_data['United States'].shift(-1)
housing_data.dropna(inplace=True)
housing_data['label'] = list(map(create_labels,housing_data['United States'], housing_data['US_HPI_future']))

X = np.array(housing_data.drop(['label','US_HPI_future'], 1))
X = preprocessing.scale(X)

y = np.array(housing_data['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
