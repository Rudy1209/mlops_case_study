from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pickle

X, y = load_iris(return_X_y=True)
model = DecisionTreeClassifier()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)