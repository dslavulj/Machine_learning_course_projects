import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

dataset = pandas.read_csv(url, names=names)

print("\nNumber of rows and columns in dataset")
print(dataset.shape)

print("\nDataset")
print(dataset.head(10))

print("\nMean,medium,max")
print(dataset.describe())

print("\nNumber of types of data")
print(dataset.groupby('class').size())


array = dataset.values
x = array[:,0:4]
y = array[:,4]

#train 70%, validate 30%
validation_size = 0.3

seed = 7

x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size = validation_size, random_state = seed)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
predictions = knn.predict(x_validation)

print(accuracy_score(y_validation, predictions))
print(classification_report(y_validation, predictions))
