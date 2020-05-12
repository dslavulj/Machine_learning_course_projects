import pandas
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

data = pandas.read_csv(url, names=names)

data.plot(kind='box',subplots = 'True', layout=(2,3), sharex=False)
#plt.show()

data.hist()
#plt.show()

scatter_matrix(data)
plt.show()