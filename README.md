# R vs Python



> (syntax ONLY, no winners here)

A repo comparing syntax in R and Python for various tasks. Not comprehensive, but a subset of lines to get one started

This is essentially a fork of a slide deck from [Decision Stats](http://www.slideshare.net/ajayohri/python-for-r-users)

More geared for R users, trying out Python than otherwise. We use Rstudio and Rmarkdown to create the reference.

> RStudio users, you may want to check out anaconda and Spyder

```
# Let us use conda to get all the packs we need
conda install pandas
```


```
## [1] "/Users/sahilseth/anaconda/bin:/Users/sahilseth/anaconda/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/X11/bin:/Library/TeX/texbin:/usr/texbin:~/anaconda/bin/:/Users/sahilseth/anaconda/bin"
```


```r
install.packages(c("e1071", "kknn", "randomForest", "rpart"))

# extra libs to compile this document
devtools::install_github("yihui/runr")
```





**Resources**:
  
  - A cheatsheet comparing R/Matlab and Python:
  http://mathesaurus.sourceforge.net/matlab-python-xref.pdf
- A book with various examples: [Machine Learning: An Algorithmic Perspective](http://www.amazon.com/gp/product/1420067184?ie=UTF8&tag=quanfinacodei-20&linkCode=as2&camp=1789&creative=9325&creativeASIN=1420067184)
- A quick how to for by [Data Robot](http://www.datarobot.com/blog/introduction-to-python-for-statistical-learning/)
A awesome slidedeck describing Python for R users


# Basic functions

Functions | R | Python
|:---|:---|:---|
Downloading and installing a package | install.packages('name') | pip install name
Load a package | library('name') | import name as other_name
Checking working directory | getwd() | import os;os.getcwd()
Setting working directory |setwd() | os.chdir()
List files in a directory |dir() | os.listdir()
List all objects  | ls() | globals()
Remove an object  | rm('name')  | del('object')

# Data Frame

## Creation

**R**
  
  Creating a data frame df of dimension 6x4 (6 rows and 4 columns) containing random
numbers


```r
A <- matrix(runif(24,0,1), nrow=6, ncol=4)
df <- data.frame(A)
print(df)
```

```
##           X1         X2        X3        X4
## 1 0.01936265 0.28521997 0.8713807 0.6103073
## 2 0.75989548 0.02908676 0.8715244 0.6717001
## 3 0.56356359 0.67548734 0.1112149 0.7882679
## 4 0.65357634 0.86374888 0.2054156 0.2614342
## 5 0.44007293 0.13416091 0.7224302 0.3420280
## 6 0.58476436 0.72279786 0.0595501 0.1235901
```


Here,
- runif function generates 24 random
numbers between 0 to 1
- matrix function creates a matrix from
those random numbers, nrow and ncol sets the numbers of rows and columns to the matrix
- data.frame converts the matrix to data frame | (Using pandas package*)


**Python**


```python
import numpy as np
import pandas as pd
A=np.random.randn(6,4)
df=pd.DataFrame(A)
print(df)
```

```
##           0         1         2         3
## 0 -1.078331 -0.814283  0.490614 -0.473481
## 1 -1.137648  1.825572  1.387178 -0.491758
## 2  1.336159 -0.532506 -2.633415 -0.413377
## 3  1.341862  0.286481 -0.757951 -0.070875
## 4 -0.269659  1.363134 -0.186930  0.672818
## 5  1.325934 -0.133586  1.097086 -2.038539
```


Here,

-  np.random.randn generates a matrix of 6 rows and 4 columns; this function is a part of `numpy` library
- pd.DataFrame converts the matrix in to a data frame


## Inspecting and Viewing Data R/Python

## Data.Frame Attributes


function | R | Python
|:---|:---|:---|
number of rows | `rownames(df)` | `df.index`
number of coliumns | `colnames(df)` | `df.index`
first few rows | `head(df)` | `df.head`
last few rows | `tail(df)` | `df.tail`
get dimensions| `dim(df)` | `df.shape`
length of df | `length(df)` | `df.len`
same as number of columns |  |

## data.frame Summary

function | R | Python
|:---|:---|:---|
quick summary including mean/std. dev etc | `summary(df)` | `df.describe`
setting row and column names | `rownames(df) = c("a", "b")` <br> `colnames(df) = c("x", "y")`| `df.index = ["a", "b"]` <br> `df.columns = ["x", "y"]`

## data.frame sorting data

function | R | Python
|:---|:---|:---|
sorting the data  | `df[order(df$x)]` | `df.sort(['x'])`

## data.frame selection

function | R | Python
|:---|:---|:---|
slicing a set of rows, from row number x to y  | `df[x:y, ]` | `df[x-1:y]` <br> Python starts counting from 0
slicing a column names  | `df[, "a"]` <br> `df$a` <br> `df["a"]` | `df.loc[:, ['a']]`
slicing a column and rows  | `df[x:y, x:y]`  | `df.iloc[x-1:y, a-1,b]`
extract specific element |  `df[x, y]`  | `df.iloc[x-1, y-1]`

## data.frame filtering/subsetting

function | R | Python
|:---|:---|:---|
subset rows where x>5 | `subset(df, x>5)` | `df[df.A> 5]`


# Math functions

function | R | Python
|:---|:---|:---|
sum | `sum(x)` | `math.fsum(x)`
square root | `sqrt(x)` | `math.sqrt(x)`
standard deviation | `sd(x)` | `numpy.std(x)`
log | `log(x)` | `math.log(x)`
mean | `mean(x)` | `numpy.mean(x)`
median | `median(x)` | `numpy.media(x)`

# Data Manipulation

function | R | Python
|:---|:---|:---|
convert character to numeric | `as.numeric(x)` | for single values: `int(x)`, `long(x)`, `float(x)` <br> for list, vectors: `map(int, x)`, `map(long, x)`, `map(float, x)`
convert numeric to character | `as.character(x)` <br> `paste(x)` | for single values: `str(x)` <br> for list, vectors: `map(str, x)`
check missing value | `is.na(x)` <br> `is.nan(x)` | `math.is.nan(x)`
remove missing value | `na.omit(x)` | [x for x in list if str(x) != 'nan']
number of chars. in value | `char(x)` | `len(x)`

## Date Manipulation

function | R (`lubridate`) | Python
|:---|:---|:---|
Getting time and date | `Sys.time()` | `d=datetime.date.time.now()`
parsing date and time: <br> `YYYY MM DD HH:MM:SS` | `lubridate::ymd_hms(Sys.time())` | `d.strftime("%Y %b %d %H:%M:%s")`

# Data Visualization

function | R | Python
|:---|:---|:---|
Scatter Plot | ` plot(variable1,variable2)`|`import matplotlib` <br> `plt.scatter(variable1,variable2);plt.show()`
Boxplot | `boxplot(Var)`|`plt.boxplot(Var);plt.show()`
Histogram | `hist(Var)` | `plt.hist(Var) plt.show()`
Pie Chart | `pie(Var)` | `from pylab import *` <br> `pie(Var) show()`

import matplotlib.pyplot as plt

Data Visualization: Bubble

# Machine Learning

## SVM on Iris Dataset

**R**

To know more about svm function in R visit: http://cran.r-project.org/web/packages/e1071/


```r
library(e1071)
data(iris)
trainset = iris[1:149,]
testset = iris[150,]
svm.model = svm(Species~., data = trainset, cost = 100, gamma = 1)
svm.pred = predict(svm.model, testset)
svm.pred
```

```
##       150 
## virginica 
## Levels: setosa versicolor virginica
```


**Python**

To install sklearn library visit [scikit-learn.org](http://scikit-learn.org)

To know more about sklearn svm visit [sklearn.svm.SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)


```
from sklearn import svm
#Importing Dataset
from sklearn import datasets
#Calling SVM
clf = svm.SVC()
#Loading the package
iris = datasets.load_iris()
# Constructing training data X,
y = iris.data[: 1], iris.target[: 1]
# Fitting SVM
clf.fit(X, y)
# Testing the model on test data print
clf.predict(iris.data[1])

# Output: Virginica Output: 2, corresponds to Virginica
```


## Linear Regression

**R**

*To know more about lm function in R visit: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/lm.html*



```r
library(broom)

data(iris)

iris$y <- sapply(as.character(iris$Species), function(x){
  switch (x,
    setosa = 0,
    versicolor = 1,
    2
  )
})

train_set <- iris[1:149,]
test_set <- iris[150,]


fit <- lm(y ~ 0+Sepal.Length+ Sepal.Width +  Petal.Length+ Petal.Width , data=train_set)
tidy(fit)
```

```
##           term    estimate  std.error  statistic      p.value
## 1 Sepal.Length -0.07454598 0.04926761 -1.5130828 1.324352e-01
## 2  Sepal.Width -0.03465755 0.05695934 -0.6084611 5.438337e-01
## 3 Petal.Length  0.21590110 0.05664803  3.8112730 2.037526e-04
## 4  Petal.Width  0.60581643 0.09340629  6.4858203 1.301553e-09
```

```r
coefficients(fit)
```

```
## Sepal.Length  Sepal.Width Petal.Length  Petal.Width 
##  -0.07454598  -0.03465755   0.21590110   0.60581643
```

```r
predict.lm(fit, test_set)
```

```
##      150 
## 1.647771
```

**Python**

*To know more about sklearn linear regression visit: http://scikit- learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html*


```python
from sklearn import linear_model
from sklearn import datasets
iris = datasets.load_iris()
regr = linear_model.LinearRegression()

X, y = iris.data[: 1], iris.target[:1]

regr.fit(X, y)

print(regr.coef_)
print(regr.predict(iris.data[1]))
```



## Random forest

**R**

*To know more about randomForest package in R visit: http://cran.r-project.org/web/packages/randomForest/*


```
library(randomForest)

iris.rf <- randomForest(y ~ .,  data=train_set,ntree=100,importance=TRUE, proximity=TRUE)

print(iris.rf)

predict(iris.rf, test_set, predict.all=TRUE)

```

**Python**

*To know more about sklearn random forest visit: http://scikit- learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html*


```python
from sklearn import ensemble
from sklearn import datasets

clf = ensemble.RandomForestClassifier(n_estimators=100, max_depth=10)
iris = datasets.load_iris()
X, y = iris.data[:1], iris.target[: 1]

clf.fit(X, y)
print(clf.predict(iris.data[1]))

# Output: 1.845 Output: 2

```


## Decision Tree

**R**

*To know more about rpart package in R visit: http://cran.r-project.org/web/packages/rpart/*



```r
library(rpart)
data(iris)

sub = c(1:149)

fit = rpart(Species ~., data = iris, subset = sub)

pred = predict(fit, iris[sub, ], type = "class")
```

**Python**

*To know more about sklearn desicion tree visit : http://scikit- learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html*


```python
from sklearn.datasets
import load_iris
from sklearn.tree
import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)
iris = datasets.load_iris()

X, y = iris.data[:1], iris.target[:1]

clf.fit(X, y)

print(clf.predict(iris.data[1]))

#Output: Virginica Output: 2, corresponds to virginica
```

## Gaussian Naive Bayes

**R**

*To know more about e1071 package in R visit: http://cran.r-project.org/web/packages/e1071/*


```r
library(e1071)
data(iris)
trainset = iris[1:149,]
testset = iris[150,]
classifier = naiveBayes(trainset[,1:4], trainset[, 5])

predict(classifier, testset[,5])
```

```
## [1] setosa
## Levels: setosa versicolor virginica
```

**Python**

*To know more about sklearn Naive Bayes visit : http://scikit- learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html*

```
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

iris = datasets.load_iris()

X, y = iris.data[:1], iris.target[:1]

clf.fit(X, y)

print(clf.predict(iris.data[1]))

#Output: Virginica Output: 2, corresponds to virginica
```


## K Nearest Neighbours

**R**

To know more about kknn package in R visit:



```r
library(kknn)
data(iris)

trainset <- iris[1:149,]
testset = iris[150,] 

iris.kknn = kknn(Species~.,  trainset,testset, distance = 1, kernel = "triangular")

summary(iris.kknn)
```

```
## 
## Call:
## kknn(formula = Species ~ ., train = trainset, test = testset,     distance = 1, kernel = "triangular")
## 
## Response: "nominal"
##         fit prob.setosa prob.versicolor prob.virginica
## 1 virginica           0        0.232759       0.767241
```

```r
fit <- fitted(iris.kknn)
fit
```

```
## [1] virginica
## Levels: setosa versicolor virginica
```

**Python**

*To know more about sklearn k nearest neighbors visit:
[scikitlearn.org](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html)*


```python
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

iris = datasets.load_iris()

X, y = iris.data[:1], iris.target[:1]

knn.fit(X,y)

print(knn.predict(iris.data[1]))

# Output: Virginica Output: 2, corresponds to virginica
```

# playing with class/objects



# writing functions


# debugging

# creating packages


# Getting help on function








