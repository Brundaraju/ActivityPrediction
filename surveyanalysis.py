import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score


def train():
    dataset = pd.read_csv('analysis_dataset.csv')
    print('Shape of the dataset: ' + str(dataset.shape))
    print(dataset.head())

    factor = pd.factorize(dataset['category'])
    dataset.category = factor[0]
    definitions = factor[1]
    print(dataset.category.head())
    print(definitions)

    X = dataset.iloc[:, 0:3].values
    y = dataset.iloc[:, 3].values
    print('The independent features set: ')
    print(X[:5, :])
    print('The dependent variable: ')
    print(y[:5])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.38, random_state=20)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.005, min_impurity_split=2,
            min_samples_leaf=70, min_samples_split=120,
            min_weight_fraction_leaf=0.0, n_estimators=60, n_jobs=-1,
            oob_score=False, random_state=20, verbose=0, warm_start=False)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    reversefactor = dict(zip(range(7), definitions))
    y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)
    print(pd.crosstab(y_test, y_pred, rownames=['Actual category'], colnames=['Predicted category']))

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print(accuracy_score(y_test, y_pred))

    print(list(zip(dataset.columns[0:3], classifier.feature_importances_)))
    joblib.dump(classifier, 'randomforestmodel.pkl')

    print(classifier)
    return scaler,reversefactor


def categorize(age,salary,budget):

    new = [age, salary, budget]
    new = np.array(new)
    new = new.reshape(1, -1)
    scaler ,reversefactor = train()
    new = scaler.fit_transform(new)
    classifier=joblib.load('randomforestmodel.pkl')
    ans = classifier.predict(new)
    ans = np.vectorize(reversefactor.get)(ans)
    return ans

def main():
    train()
    print(categorize(20,7,1000))

if __name__ == '__main__':
    main()
