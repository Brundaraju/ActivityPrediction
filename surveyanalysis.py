import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

def train():
    dataset = pd.read_csv('surveydata.csv')
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    classifier = RandomForestClassifier(n_estimators=5, criterion='entropy', random_state=42)
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


if __name__ == '__main__':
    main()