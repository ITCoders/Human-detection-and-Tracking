from time import time
import numpy
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from PIL import Image

# Path to the root image directory containing sub-directories of images
path = "/home/arpit/ATnT_data/"
testImage = "/home/arpit/new/10.pgm"

# Flat image Feature Vector
X = []
# Int array of Label Vector
Y = []

n_sample = 0  # Total number of Images
h = 112  # Height of image in float
w = 92  # Width of image in float
n_features = 187500  # Length of feature vector
target_names = []  # Array to store the names of the persons
label_count = 0
n_classes = 0

for directory in os.listdir(path):
    for file in os.listdir(path + directory):
        print(path + directory + "/" + file)
        img = Image.open(path + directory + "/" + file)
        featurevector = numpy.array(img).flatten()
        # print len(featurevector)
        X.append(featurevector)
        Y.append(label_count)
        n_sample += 1
    target_names.append(directory)
    label_count += 1

# print Y
# print target_names
n_classes = len(target_names)

###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and teststing set
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42)

###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 10

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, len(X_test)))
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

###############################################################################
# Train a SVM classification model
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

###############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
# print clf.score(X_test_pca,y_test)
print("done in %0.3fs" % (time() - t0))
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

###############################################################################

A = []
image = Image.open(testImage)
feature_vec = numpy.array(image).flatten()
A.append(feature_vec)
A_pca = pca.transform(A)
A_predict = clf.predict(A_pca)
B = numpy.array(A)
print(B)
print(A_predict)
print(classification_report(B, A_predict))
# Prediction of user based on the model
# test = []
# testImage=Image.open(testImage)
# testImageFeatureVector=numpy.array(testImage).flatten()
# test.append(testImageFeatureVector)
# testImagePCA = pca.transform(test)
# testImagePredict=clf.predict(testImagePCA)
# print(clf.score(testImagePCA))
# print(clf.score(X_train_pca,testImagePCA))
# print(clf.best_params_)
# print(clf.best_score_)
# print(testImagePredict)
# print(target_names[testImagePredict[0]])
