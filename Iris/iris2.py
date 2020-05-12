import pandas

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

data = pandas.read_csv(url, names=names)

#peak at unseen data
peek = data.head(30)
print(peek)

#dimensions of dataset
shape = data.shape
print(shape)

#type of each attribute
types = data.dtypes
print(types)

#descriptive statistics
pandas.set_option('display.width', 100)
pandas.set_option('precision', 3)
description = data.describe()
print(description)

#correlation between attributes
correlations = data.corr(method='pearson')
print(correlations)

#skew for data
skew = data.skew()
print(skew)
