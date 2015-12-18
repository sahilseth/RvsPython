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
## [1] "/Users/sahilseth/anaconda/bin:/Users/sahilseth/anaconda/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/X11/bin:/Library/TeX/texbin:/usr/texbin"
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
##            X1          X2         X3        X4
## 1 0.008414911 0.311282917 0.61186960 0.8987637
## 2 0.021930034 0.072141218 0.96503840 0.9508710
## 3 0.227929049 0.840288003 0.61093433 0.3269284
## 4 0.162879566 0.325983315 0.82753045 0.4227151
## 5 0.593558020 0.009578978 0.84678802 0.2197988
## 6 0.219805626 0.054050339 0.04714518 0.1655515
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
## 0 -0.217405 -0.163276  0.936169 -0.089373
## 1  2.276137  0.891530  1.257429 -0.686684
## 2  0.295248 -0.528968  0.364880  0.274526
## 3  0.854174 -2.911316  0.768290  0.972371
## 4 -1.377254  2.524532 -0.718311  1.294197
## 5  0.252250 -0.408106 -0.598757  1.542085
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



```python
from sklearn import svm
from sklearn import datasets

#Calling SVM
clf = svm.SVC()

iris = datasets.load_iris()

# Constructing training data X,
X, y = iris.data[:-1], iris.target[:-1]

# Fitting SVM
clf.fit(X, y)

# Testing the model on test data print
clf.predict(iris.data[-1])

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

X, y = iris.data[:-1], iris.target[:-1]

regr.fit(X, y)

print(regr.coef_)
print(regr.predict(iris.data[-1]))
```

```
## [-0.09726197 -0.05347337  0.21782359  0.61500051]
## [ 1.65708429]
```



## Random forest

**R**

*To know more about randomForest package in R visit: http://cran.r-project.org/web/packages/randomForest/*



```r
library(randomForest)
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

```r
iris.rf <- randomForest(y ~ .,  data=train_set,ntree=100,importance=TRUE, proximity=TRUE)
```

```
## Warning in randomForest.default(m, y, ...): The response has five or fewer
## unique values. Are you sure you want to do regression?
```

```r
print(iris.rf)
```

```
## 
## Call:
##  randomForest(formula = y ~ ., data = train_set, ntree = 100,      importance = TRUE, proximity = TRUE) 
##                Type of random forest: regression
##                      Number of trees: 100
## No. of variables tried at each split: 1
## 
##           Mean of squared residuals: 0.01151123
##                     % Var explained: 98.27
```

```r
predict(iris.rf, test_set, predict.all=TRUE)
```

```
## $aggregate
##      150 
## 1.963167 
## 
## $individual
##     [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12] [,13]
## 150    2    2    2    2    2    2    2    2    2     2     2     2     2
##     [,14] [,15] [,16] [,17] [,18] [,19] [,20] [,21] [,22] [,23] [,24]
## 150     2   1.8     2     2     2     2     2     2     2     2     2
##     [,25] [,26] [,27] [,28] [,29] [,30] [,31] [,32] [,33] [,34] [,35]
## 150     2     2     2     2     2     2     2     2     2     2     2
##     [,36] [,37] [,38] [,39] [,40] [,41] [,42] [,43] [,44] [,45] [,46]
## 150     2     2     2     2     2   1.8     2     2     2     2     2
##     [,47] [,48] [,49] [,50] [,51] [,52] [,53] [,54] [,55] [,56] [,57]
## 150     2     2     2     2     2     2     1     2     2     2     2
##     [,58] [,59] [,60] [,61] [,62] [,63] [,64] [,65] [,66] [,67] [,68]
## 150     2     2     2     2     2     2     2     2     2     2     2
##     [,69] [,70] [,71] [,72] [,73] [,74] [,75] [,76] [,77] [,78] [,79]
## 150     2     2   1.8     2     2     2     2     2     2     2     2
##     [,80] [,81] [,82] [,83] [,84] [,85] [,86] [,87] [,88] [,89]    [,90]
## 150     2     2     2     2     2     1     2     2     2     2 1.666667
##     [,91] [,92] [,93] [,94] [,95] [,96] [,97] [,98] [,99] [,100]
## 150     2     2     2     2  1.25     2     2     2     2      2
```

**Python**

*To know more about sklearn random forest visit: http://scikit- learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html*


```python
from sklearn import ensemble
from sklearn import datasets

clf = ensemble.RandomForestClassifier(n_estimators=100, max_depth=10)
iris = datasets.load_iris()
X, y = iris.data[:-1], iris.target[:-1]

clf.fit(X, y)
print(clf.predict(iris.data[-1]))

# Output: 1.845 Output: 2

```

```
## [2]
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
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()

X, y = iris.data[:-1], iris.target[:-1]

clf.fit(X, y)

print(clf.predict(iris.data[-1]))

#Output: Virginica Output: 2, corresponds to virginica
```

```
## [2]
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


```python
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
iris = load_iris()
X, y = iris.data[:-1], iris.target[:-1]
clf.fit(X, y)
print(clf.predict(iris.data[-1]))
#Output: Virginica Output: 2, corresponds to virginica
```

```
## [2]
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

iris = load_iris()

X, y = iris.data[:-1], iris.target[:-1]

knn.fit(X,y)

print(knn.predict(iris.data[-1]))

# Output: Virginica Output: 2, corresponds to virginica
```

# playing with class/objects



# writing functions


# debugging

# creating packages


# Getting help on function





```r
#py$stop()
```






