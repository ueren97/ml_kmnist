from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def load(f):
    return np.load(f)['arr_0']


# Load the data
X_train = load('../dataset/kmnist-train-imgs.npz')
X_test = load('../dataset/kmnist-test-imgs.npz')
y_train = load('../dataset/kmnist-train-labels.npz')
y_test = load('../dataset/kmnist-test-labels.npz')

# Flatten images
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

clf = KNeighborsClassifier(n_neighbors=4, weights='distance', n_jobs=-1)
print('Fitting', clf)
clf.fit(X_train, y_train)
print('Evaluating', clf)

test_score = clf.score(X_test, y_test)
print('Test accuracy:', test_score)
