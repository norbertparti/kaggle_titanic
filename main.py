import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import precision_score

from plotting_utils import plot_correlation

# Load in the train and test datasets
train = pd.read_csv('./input/train.csv')
train = train.dropna()
#test = pd.read_csv('./input/test.csv')
used_columns = ['Pclass', 'Sex', 'Age' , 'Parch', 'Fare', 'Embarked']


#Transform train and test data
filtered_train = train[used_columns]

filtered_train['Sex'] = filtered_train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
filtered_train['Embarked'] = filtered_train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

min_max_scaler = preprocessing.MinMaxScaler()
scaled_values = min_max_scaler.fit_transform(filtered_train.values)
x_scaled = pd.DataFrame(scaled_values, columns=used_columns)

_true = train['Survived']
#filtered_test = test[used_columns_test].dropna()
#filtered_test['Sex'] = filtered_test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
#filtered_test['Embarked'] = filtered_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# train test split
X_train, X_test, y_train, y_test = train_test_split(x_scaled, _true, test_size=0.33, random_state=42)


to_count_embarked = filtered_train.Embarked

number_of_categories_embarked = len(to_count_embarked.value_counts())


plot_correlation(x_scaled)


gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

prec_score = precision_score(y_true=y_test, y_pred=y_pred)

print('Precision {}'.format(prec_score))


