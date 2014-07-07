---

title       : Machine Learning with R
author      : Ilan Man
job         : Strategy Operations  @ Squarespace
framework   : io2012        # {io2012, html5slides, shower, dzslides, ...}
highlighter : highlight.js  # {highlight.js, prettify, highlight}
hitheme     : tomorrow      # 
widgets     : mathjax       # {mathjax, quiz, bootstrap}
mode        : selfcontained # {standalone, draft}

----

## Agenda 
<space>

1. Machine Learning Overview
2. Exploring Data
3. Nearest Neighbors
4. Naive Bayes
5. Measuring Performance
6. Linear Regression

----

## Machine Learning Overview
# What is it?  
<space>

- Field of study interested in transforming data into intelligent actions
- Intersection of statistics, available data and computing power
- It is NOT data mining
- Data mining is an exploratory exercise, whereas most machine learning has a known answer
- Data mining is a subset of machine learning (unsupervised)

----

## Machine Learning Overview
# Uses
<space>

- Predict outcome of elections
- Email filtering - spam or not
- Credit fraud prediction
- Image processing
- Customer churn
- Customer subscription rates

----

## Machine Learning Overview
# How do machines learn?
<space>

- Data input 
  - Provides a factual basis for reasoning
- Abstraction 
- Generalization 

----

## Machine Learning Overview
# Abstraction
<space>

- Assign meaning to the data
- Formulas, graphs, logic, etc...
- Your model
- Fitting model is called training

----

## Machine Learning Overview
# Generalization
<space>

- Turn abstracted knowledge into something that can be utilized
- Model user heuristics since it cannot see every example
  - When hueristics are systematically wrong, the algorithm has a bias
- Very simple models have high bias
  - Some bias is good - let's us ignore the noise

----

## Machine Learning Overview
# Generalization
<space>

- After training, the model is tested on unseen data
- Perfect generalization is exceedingly rare
  - Partly due to noise
  - Measurement error
  - Change in user behavior
  - Incorrect data, erroneous values, etc...
- Fitting too closesly to the noise leads to overfitting
  - Complex models have high variance
  - Good on training, bad on testing

----

## Machine Learning Overview
# Steps to apply Machine Learning
<space>

1. Collect data
2. Explore and preprocess data 
  - Majority of the time is spent in this stage
3. Train the model
  - Specific tasks will inform which algorithm is appropriate
4. Evaluate model performance
  - Performance measures depend on use case
5. Improve model performance as necessary

----

## Machine Learning Overview
# Choosing an algorithm
<space>

- Consider input data
- An <strong>example</strong> is one data point that the machine is intended to learn 
- A feature is a characteristic of the example
  - e.g. Number of times the word "viagra" appears in an email
- For classification problems, a label is the example's classification
- Most algorithms require data in matrix format because Math said so
- Features can be numeric, categorical/nominal or ordinal

----

## Machine Learning Overview
# Types of algorithms
<space>

- Supervised
  - Discover relationship between known, target feature and other features
  - Predictive
  - Classification and numeric prediction tasks
- Unsupervised
  - Unkown answer
  - Descriptive
  - Pattern discovery and clustering into groups
  - Requires human intervention to interpret clusters

----

## Machine Learning Overview
# Summary
<space>

1. Generalization and Abstraction
2. Overfitting vs underfitting
3. The right algorithm will be informed by the problem to be solved
4. Terminology

----

## Exploring Data
# Exploring and understanding data
<space>

- Load and explore the data


```r
data(iris)

# inspect the structure of the dataset
str(iris)
```

```
## 'data.frame':	150 obs. of  5 variables:
##  $ Sepal.Length: num  5.1 4.9 4.7 4.6 5 5.4 4.6 5 4.4 4.9 ...
##  $ Sepal.Width : num  3.5 3 3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 ...
##  $ Petal.Length: num  1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 ...
##  $ Petal.Width : num  0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 ...
##  $ Species     : Factor w/ 3 levels "setosa","versicolor",..: 1 1 1 1 1 1 1 1 1 1 ...
```


----

## Exploring Data
# Exploring and understanding data
<space>


```r
# summarize the data - five number summary
summary(iris[, 1:4])
```

```
##   Sepal.Length   Sepal.Width    Petal.Length   Petal.Width 
##  Min.   :4.30   Min.   :2.00   Min.   :1.00   Min.   :0.1  
##  1st Qu.:5.10   1st Qu.:2.80   1st Qu.:1.60   1st Qu.:0.3  
##  Median :5.80   Median :3.00   Median :4.35   Median :1.3  
##  Mean   :5.84   Mean   :3.06   Mean   :3.76   Mean   :1.2  
##  3rd Qu.:6.40   3rd Qu.:3.30   3rd Qu.:5.10   3rd Qu.:1.8  
##  Max.   :7.90   Max.   :4.40   Max.   :6.90   Max.   :2.5
```


----

## Exploring Data
# Exploring and understanding data
<space>

- Measures of central tendency: mean and median
  - Mean is sensitive to outliers
    - Trimmed mean
  - Median is resistant

![plot of chunk skew](figure/skew.png) 


----

## Exploring Data
# Exploring and understanding data
<space>

- Measures of dispersion
  - Range is the `max()` - `min()`
  - Interquartile range (IQR) is the `Q3` - `Q1`
  - Quantile


```r
quantile(iris$Sepal.Length, probs = c(0.1, 0.5, 0.99))
```

```
10% 50% 99% 
4.8 5.8 7.7 
```


----

## Exploring Data
# Visualizing - Boxplots
<space>

- Lets you see the spread in the data

<img src="figure/boxplot.png" title="plot of chunk boxplot" alt="plot of chunk boxplot" style="display: block; margin: auto;" />


----

## Exploring Data
# Visualizing - histograms
<space>

- Each bar is a 'bin'
- Height of bar is the frequency (count of) that bin
- Some distributions are normally distributed (bell shaped) or skewed (heavy tails)

<img src="figure/histogram.png" title="plot of chunk histogram" alt="plot of chunk histogram" style="display: block; margin: auto;" />


----

## Exploring Data
# Visualizing - scatterplots
<space>

- Useful for visualizing bivariate relationships (2 variables)

<img src="figure/scatterplot.png" title="plot of chunk scatterplot" alt="plot of chunk scatterplot" style="display: block; margin: auto;" />


----

## Exploring Data
# Summary
<space>

- Measures of central tendency and dispersion
- Visualizing data using histograms, boxplots, scatterplots
- Skewed vs normally distributed data

----

## K-Nearest Neighbors
# Classification using kNN
<space>

- Understanding the algorithm
- Data Preparation
- Case study: diagnosing breast cancer
- Summary

----

## K-Nearest Neighbors
# The Concept
<space>

- Things that are similar are probably of the same class
- Good for: when it's difficult to define, but "you know it when you see it"
- Bad for: when a clear distinction doesn't exist

----

## K-Nearest Neighbors
# The Algorithm
<space>

![plot of chunk knn_polygon_plot](figure/knn_polygon_plot.png) 


----

## K-Nearest Neighbors
# The Algorithm
<space>

![plot of chunk shaded plot](figure/shaded_plot.png) 


----

## K-Nearest Neighbors
# The Algorithm
<space>

<img src="figure/new_point.png" title="plot of chunk new point" alt="plot of chunk new point" style="display: block; margin: auto auto auto 0;" />


- Suppose we had a new point with Sepal Length of 7 and Petal Length of 4
- Which species will it probably belong to?

----

## K-Nearest Neighbors
# The Algorithm
<space>

- Calculate its nearest neighbor
  - Euclidean distance
  - $dist(p,q) = \sqrt{(p_1-q_1)^2+(p_2-q_2)^2+ ... + (p_n-q_n)^2}$
  - Closest neighbor -> 1-NN
  - 3 closest neighbors -> 3-NN. 
  - Winner is the majority class of all neighbors

----

## K-Nearest Neighbors
# The Algorithm
<space>

- Calculate its nearest neighbor
  - Euclidean distance
  - $dist(p,q) = \sqrt{(p_1-q_1)^2+(p_2-q_2)^2+ ... + (p_n-q_n)^2}$
  - Closest neighbor -> 1-NN
  - 3 closest neighbors -> 3-NN. 
  - Winner is the majority class of all neighbors
- Why not just fit to all data points?

----

## K-Nearest neighbors
# Bias vs. Variance
<space>

- Fitting to every point results in an overfit model
  - High variance problem
- Fitting to only 1 point results in an underfit model
  - High bias problem
- Choosing the right $k$ is a balance between bias and variance
- Rule of thumb: $k = \sqrt{N}$

----

## K-Nearest neighbors
# Data preparation
<space>

- Classify houses based on prices and square footage


```r
library(scales)  # format ggplot() axis
price <- seq(3e+05, 6e+05, by = 10000)
size <- price/1000 + rnorm(length(price), 10, 50)
houses <- data.frame(price, size)
ex <- ggplot(houses, aes(price, size)) + geom_point() + scale_x_continuous(labels = comma) + 
    xlab("Price") + ylab("Size") + ggtitle("Square footage vs Price")
```


----

## K-Nearest neighbors
# Data Preparation
<space>

![plot of chunk house price plot](figure/house_price_plot.png) 


----

## K-Nearest neighbors
# Data Preparation
<space>

![plot of chunk clusters](figure/clusters.png) 


----

## K-Nearest neighbors
# Data Preparation
<space>

![plot of chunk new cluster plot](figure/new_cluster_plot.png) 


----

## K-Nearest neighbors
# Data Preparation
<space>


```r
# 1) using loops
loop_dist <- 0
for (i in 1:nrow(houses)) {
    loop_dist[i] <- sqrt(sum((new_p - houses[i, ])^2))
}

# 2) vectorized
vec_dist <- sqrt(rowSums(t(new_p - t(houses))^2))
closest <- data.frame(houses[which.min(vec_dist), ])
print(closest)
```

```
   price  size
11 4e+05 323.8
```


----

## K-Nearest Neighbors
# Data Preparation
<space>

![plot of chunk knn_repeat_plot](figure/knn_repeat_plot.png) 


----

## K-Nearest Neighbors
# Data Preparation
<space>

![plot of chunk closest_knn](figure/closest_knn.png) 


----

## K-Nearest Neighbors
# Data Preparation
<space>

- Feature scaling. Level the playing field.
- Two common approaches:
  - min-max normalization
    - $X_{new} = \frac{X-min(X)}{max(X) - min(X)}$
  - z-score standardization
    - $X_{new} = \frac{X-mean(X)}{sd(X)}$
- Euclidean distance doesn't discriminate between important and noisy features
  - can add weights

----

## K-Nearest Neighbors
# Data Preparation
<space>


```r
new_house <- scale(houses)
new_new <- c((new_p[1] - mean(houses[, 1]))/sd(houses[, 1]), (new_p[2] - mean(houses[, 
    2]))/sd(houses[, 2]))

vec_dist <- sqrt(rowSums(t(new_new - t(new_house))^2))
scale_closest <- data.frame(houses[which.min(vec_dist), ])
```


----

## K-Nearest Neighbors
# Data Preparation
<space>

![plot of chunk scale_closest_plot](figure/scale_closest_plot.png) 


----

## K-Nearest Neighbors
# Case study
<space>


```r
data <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", 
    sep = ",", stringsAsFactors = FALSE, header = FALSE)

# first column has the ID which is not useful
data <- data[, -1]

# names taken from the .names file online
n <- c("radius", "texture", "perimeter", "area", "smoothness", "compactness", 
    "concavity", "concave_points", "symmetry", "fractal")
ind <- c("mean", "std", "worst")

headers <- as.character()
for (i in ind) {
    headers <- c(headers, paste(n, i))
}
names(data) <- c("diagnosis", headers)
```


----

## K-Nearest Neighbors
# Case study
<space>


```r
str(data[, 1:10])
```

```
'data.frame':	569 obs. of  10 variables:
 $ diagnosis          : chr  "M" "M" "M" "M" ...
 $ radius mean        : num  18 20.6 19.7 11.4 20.3 ...
 $ texture mean       : num  10.4 17.8 21.2 20.4 14.3 ...
 $ perimeter mean     : num  122.8 132.9 130 77.6 135.1 ...
 $ area mean          : num  1001 1326 1203 386 1297 ...
 $ smoothness mean    : num  0.1184 0.0847 0.1096 0.1425 0.1003 ...
 $ compactness mean   : num  0.2776 0.0786 0.1599 0.2839 0.1328 ...
 $ concavity mean     : num  0.3001 0.0869 0.1974 0.2414 0.198 ...
 $ concave_points mean: num  0.1471 0.0702 0.1279 0.1052 0.1043 ...
 $ symmetry mean      : num  0.242 0.181 0.207 0.26 0.181 ...
```


----

## K-Nearest Neighbors
# Case study
<space>


```r
prop.table(table(data$diagnosis))  # Balanced data set?
```

```

     B      M 
0.6274 0.3726 
```

```r
head(data)[2:6]  # inspect features more closely
```

```
  radius mean texture mean perimeter mean area mean smoothness mean
1       17.99        10.38         122.80    1001.0         0.11840
2       20.57        17.77         132.90    1326.0         0.08474
3       19.69        21.25         130.00    1203.0         0.10960
4       11.42        20.38          77.58     386.1         0.14250
5       20.29        14.34         135.10    1297.0         0.10030
6       12.45        15.70          82.57     477.1         0.12780
```


----

## K-Nearest Neighbors
# Case study
<space>


```r
# scale each numeric value
scaled_data <- as.data.frame(lapply(data[, -1], scale))
scaled_data <- cbind(diagnosis = data$diagnosis, scaled_data)
head(scaled_data[2:6])
```

```
  radius.mean texture.mean perimeter.mean area.mean smoothness.mean
1      1.0961      -2.0715         1.2688    0.9835          1.5671
2      1.8282      -0.3533         1.6845    1.9070         -0.8262
3      1.5785       0.4558         1.5651    1.5575          0.9414
4     -0.7682       0.2535        -0.5922   -0.7638          3.2807
5      1.7488      -1.1508         1.7750    1.8246          0.2801
6     -0.4760      -0.8346        -0.3868   -0.5052          2.2355
```


----

## K-Nearest Neighbors
# Case study
<space>


```r
library(class)  # get k-NN classifier
predict_1 <- knn(train = scaled_data[, 2:31], test = scaled_data[, 2:31], cl = scaled_data[, 
    1], k = floor(sqrt(nrow(scaled_data))))
prop.table(table(predict_1))
```

```
predict_1
     B      M 
0.6643 0.3357 
```


----

## K-Nearest Neighbors
# Case study
<space>


```r
pred_B <- which(predict_1 == "B")
actual_B <- which(scaled_data[, 1] == "B")
pred_M <- which(predict_1 == "M")
actual_M <- which(scaled_data[, 1] == "M")

true_positive <- sum(pred_B %in% actual_B)
true_negative <- sum(pred_M %in% actual_M)
false_positive <- sum(pred_B %in% actual_M)
false_negative <- sum(pred_M %in% actual_B)

conf_mat <- matrix(c(true_positive, false_positive, false_negative, true_negative), 
    nrow = 2, ncol = 2)

acc <- sum(diag(conf_mat))/sum(conf_mat)
tpr <- conf_mat[1, 1]/sum(conf_mat[1, ])
tn <- conf_mat[2, 2]/sum(conf_mat[2, ])
```


----

## K-Nearest Neighbors
# Case study
<space>


```
   acc    tpr     tn 
0.9596 0.9972 0.8962 
```

```
       Actual B Actual M
Pred B      356        1
Pred M       22      190
```


- Is that right?

----

## K-Nearest Neighbors
# Case study
<space>


```r
# create randomized training and testing sets
total_n <- nrow(scaled_data)

# train on 2/3 of the data
train_ind <- sample(total_n, total_n * 2/3)

train_labels <- scaled_data[train_ind, 1]
test_labels <- scaled_data[-train_ind, 1]

train_set <- scaled_data[train_ind, 2:31]
test_set <- scaled_data[-train_ind, 2:31]
```


----

## K-Nearest Neighbors
# Case study
<space>


```r
predict_1 <- knn(train = train_set, test = test_set, cl = train_labels, k = floor(sqrt(nrow(train_set))))
prop.table(table(predict_1))
```

```
predict_1
     B      M 
0.6789 0.3211 
```


----

## K-Nearest Neighbors
# Case study
<space>


```r
pred_B <- which(predict_1 == "B")
test_B <- which(test_labels == "B")
pred_M <- which(predict_1 == "M")
test_M <- which(test_labels == "M")

true_positive <- sum(pred_B %in% test_B)
true_negative <- sum(pred_M %in% test_M)
false_positive <- sum(pred_B %in% test_M)
false_negative <- sum(pred_M %in% test_B)

conf_mat <- matrix(c(true_positive, false_negative, false_positive, true_negative), 
    nrow = 2, ncol = 2)

acc <- sum(diag(conf_mat))/sum(conf_mat)
tpr <- conf_mat[1, 1]/sum(conf_mat[1, ])
tn <- conf_mat[2, 2]/sum(conf_mat[2, ])
```


----

## K-Nearest Neighbors
# Case study
<space>


```
   acc    tpr     tn 
0.9579 0.9380 1.0000 
```

```
       Actual B Actual M
Pred B      121        8
Pred M        0       61
```


----

## K-Nearest Neighbors
# Case study
<space>


```r
library(caret)  # Classification and Regression Training
con_mat <- confusionMatrix(predict_1, test_labels)
con_mat$table
```

```
          Reference
Prediction   B   M
         B 121   8
         M   0  61
```



----

## K-Nearest Neighbors
# Case study
<space>


```r
library(caret)  # Classification and Regression Training
con_mat <- confusionMatrix(predict_1, test_labels)
con_mat$table
```

```
          Reference
Prediction   B   M
         B 121   8
         M   0  61
```


```
Exercise:
1) Find the Accuracy for various values of k. What's the best value of k for your model?
```

----

## K-Nearest Neighbors
# Case study
<space>


```r
k_params <- c(1, 3, 5, 10, 15, 20, 25, 30, 40)
perf_acc <- NULL
per <- NULL
for (i in k_params) {
    predictions <- knn(train = train_set, test = test_set, cl = train_labels, 
        k = i)
    conf <- confusionMatrix(predictions, test_labels)$table
    perf_acc <- sum(diag(conf))/sum(conf)
    per <- rbind(per, c(i, perf_acc, conf[[1]], conf[[3]], conf[[2]], conf[[4]]))
}
```


----

## K-Nearest Neighbors
# Case study
<space>

<!-- html table generated in R 3.0.3 by xtable 1.7-3 package -->
<!-- Mon Jul 07 19:28:56 2014 -->
<TABLE border=1>
<TR> <TH> K </TH> <TH> Acc </TH> <TH> TP </TH> <TH> FP </TH> <TH> FN </TH> <TH> TN </TH>  </TR>
  <TR> <TD align="right"> 1.00 </TD> <TD align="right"> 0.97 </TD> <TD align="right"> 118.00 </TD> <TD align="right"> 2.00 </TD> <TD align="right"> 3.00 </TD> <TD align="right"> 67.00 </TD> </TR>
  <TR> <TD align="right"> 3.00 </TD> <TD align="right"> 0.97 </TD> <TD align="right"> 120.00 </TD> <TD align="right"> 5.00 </TD> <TD align="right"> 1.00 </TD> <TD align="right"> 64.00 </TD> </TR>
  <TR> <TD align="right"> 5.00 </TD> <TD align="right"> 0.97 </TD> <TD align="right"> 121.00 </TD> <TD align="right"> 5.00 </TD> <TD align="right"> 0.00 </TD> <TD align="right"> 64.00 </TD> </TR>
  <TR> <TD align="right"> 10.00 </TD> <TD align="right"> 0.96 </TD> <TD align="right"> 121.00 </TD> <TD align="right"> 8.00 </TD> <TD align="right"> 0.00 </TD> <TD align="right"> 61.00 </TD> </TR>
  <TR> <TD align="right"> 15.00 </TD> <TD align="right"> 0.96 </TD> <TD align="right"> 121.00 </TD> <TD align="right"> 8.00 </TD> <TD align="right"> 0.00 </TD> <TD align="right"> 61.00 </TD> </TR>
  <TR> <TD align="right"> 20.00 </TD> <TD align="right"> 0.95 </TD> <TD align="right"> 121.00 </TD> <TD align="right"> 9.00 </TD> <TD align="right"> 0.00 </TD> <TD align="right"> 60.00 </TD> </TR>
  <TR> <TD align="right"> 25.00 </TD> <TD align="right"> 0.96 </TD> <TD align="right"> 121.00 </TD> <TD align="right"> 8.00 </TD> <TD align="right"> 0.00 </TD> <TD align="right"> 61.00 </TD> </TR>
  <TR> <TD align="right"> 30.00 </TD> <TD align="right"> 0.96 </TD> <TD align="right"> 121.00 </TD> <TD align="right"> 8.00 </TD> <TD align="right"> 0.00 </TD> <TD align="right"> 61.00 </TD> </TR>
  <TR> <TD align="right"> 40.00 </TD> <TD align="right"> 0.95 </TD> <TD align="right"> 121.00 </TD> <TD align="right"> 9.00 </TD> <TD align="right"> 0.00 </TD> <TD align="right"> 60.00 </TD> </TR>
   </TABLE>


----

## K-Nearest Neighbors
# Summary
<space>
  
- kNN is a lazy learning algorithm
  - Stores training data and applies it verbatim to new data
  - "instance-based" learning
- Assigns the majority class of the k data points closest to the new data
  - Ensure all features are on the same scale
- Strengths
  - Can be applied to data from any distribution
  - Simple and intuitive
- Weaknesses
  - Choosing k requires trial and error
  - Testing step is computationally expensive (unlike parametric models)
  - Needs a large number of training samples to be useful

----

## Naive Bayes
# Probabilistic learning 
<space>

- Probability and Bayes Theorem
- Understanding Naive Bayes
- Case study: filtering mobile phone spam

----

## Naive Bayes
# Probability and Bayes Theorem
<space>

- Terminology: 
  + `probability`
  + `event` 
  + `trial` - e.g. 1 flip of a coin, 1 toss of a die
- $X_{i}$ is an event
- The set of all events is $\{X_{1},X_{2},...,X_{n}\}$
- The probability of an event is the frequency of its occurrence
  + $0 \leq P(X) \leq 1$
  + $P(\sum_{i=1}^{n} X_{i}) = \sum_{i=1}^{n} P(X_{i})$   (for mutually excluse $X_{i}$)

----

## Naive Bayes
# Probability and Bayes Theorem
<space>

- "A and B" is $A \cap B$ (intersect)
- Independent events
  - $P(A \cap B) = P(A) \times P(B)$
<space>
- "A given B" is $A \mid B$
- Conditional probability
  - $P(A \mid B) = \frac{P(A \cap B)}{P(B)}$

----

## Naive Bayes
# Probability and Bayes Theorem
<space>

- "A and B" is $A \cap B$ (intersect)
- Independent events
  - $P(A \cap B) = P(A) \times P(B)$
<space>
- "A given B" is $A \mid B$
- Conditional probability
  - $P(A \mid B) = \frac{P(A \cap B)}{P(B)}$ 
  - $P(B \mid A) = \frac{P(B \cap A)}{P(A)}$

----

## Naive Bayes
# Probability and Bayes Theorem
<space>

- "A and B" is $A \cap B$ (intersect)
- Independent events
  - $P(A \cap B) = P(A) \times P(B)$
<space>
- "A given B" is $A \mid B$
- Conditional probability
  - $P(A \mid B) = \frac{P(A \cap B)}{P(B)}$
  - $P(B \mid A) = \frac{P(B \cap A)}{P(A)}$
  - $P(B \mid A) \times P(A) = P(B \cap A)$

----

## Naive Bayes
# Probability and Bayes Theorem
<space>

- "A and B" is $A \cap B$ (intersect)
- Independent events
  - $P(A \cap B) = P(A) \times P(B)$
<space>
- "A given B" is $A \mid B$
- Conditional probability
  - $P(A \mid B) = \frac{P(A \cap B)}{P(B)}$
  - $P(B \mid A) = \frac{P(B \cap A)}{P(A)}$
  - $P(B \mid A) \times P(A) = P(B \cap A)$
  + but $P(B \cap A) = P(A \cap B)$

----

## Naive Bayes
# Probability and Bayes Theorem
<space>

- "A and B" is $A \cap B$ (intersect)
- Independent events
  - $P(A \cap B) = P(A) \times P(B)$
<space>
- "A given B" is $A \mid B$
- Conditional probability
  - $P(A \mid B) = \frac{P(A \cap B)}{P(B)}$
  - $P(B \mid A) = \frac{P(B \cap A)}{P(A)}$
  - $P(B \mid A) \times P(A) = P(B \cap A)$
  + but $P(B \cap A) = P(A \cap B)$
  + so $P(A \mid B) = \frac{P(B \mid A) \times P(A)}{P(B)}$

----

## Naive Bayes
# Probability and Bayes Theorem
<space>

- "A and B" is $A \cap B$ (intersect)
- Independent events
  - $P(A \cap B) = P(A) \times P(B)$
<space>
- "A given B" is $A \mid B$
- Conditional probability
  - $P(A \mid B) = \frac{P(A \cap B)}{P(B)}$
  - $P(B \mid A) = \frac{P(B \cap A)}{P(A)}$
  - $P(B \mid A) \times P(A) = P(B \cap A)$
  + but $P(B \cap A) = P(A \cap B)$
  + so $P(A \mid B) = \frac{P(B \mid A) \times P(A)}{P(B)}$ &nbsp;&nbsp;<-  &nbsp; Bayes theorem

----

## Naive Bayes
# Bayes Example
<space>

- A decision should be made using all available information
  - As new information enters, the decision might be changed
- Example: Email filtering
  - spam and non-spam (AKA ham)
  - classify emails depending on what words they contain
  - spam emails are more likely to contain certain words
  - $P(spam \mid CASH!)$

----

## Naive Bayes
# Bayes Example
<space>


```r
# data frame with frequency of emails with the word 'cash'
bayes_ex <- data.frame(cash_yes = c(10, 3, 13), cash_no = c(20, 67, 87), total = c(30, 
    70, 100), row.names = c("spam", "ham", "total"))
bayes_ex
```

```
      cash_yes cash_no total
spam        10      20    30
ham          3      67    70
total       13      87   100
```


----

## Naive Bayes
# Bayes Example
<space>

- Recall Bayes Theorem: 
  - $P(A \mid B) = \frac{P(B \mid A) \times P(A)}{P(B)}$
- A = event that email is spam  
- B = event that "CASH" exists in the email  
<space>
$P(spam \mid cash=yes) = P(cash=yes \mid spam) \times \frac{P(spam)}{P(cash=yes)}$

----

## Naive Bayes
# Bayes Example
<space>

- Recall Bayes Theorem: 
  - $P(A \mid B) = \frac{P(B \mid A) \times P(A)}{P(B)}$
- A = event that email is spam  
- B = event that "CASH" exists in the email  
$P(spam \mid cash=yes) = P(cash=yes \mid spam) \times \frac{P(spam)}{P(cash=yes)}$<br>
$P(cash = yes \mid spam) = \frac{10}{30}$<br>
$P(spam) =  \frac{30}{100}$<br>
$P(cash = yes) = \frac{13}{100}$<br>
 = $\frac{10}{30} \times \frac{\frac{30}{100}}{\frac{13}{100}} = 0.769$ 

----

## Naive Bayes
# Bayes Example
<space>

- Recall Bayes Theorem: 
  - $P(A \mid B) = \frac{P(B \mid A) \times P(A)}{P(B)}$
- A = event that email is spam  
- B = event that "CASH" exists in the email  
$P(spam \mid cash=yes) = P(cash=yes \mid spam) \times \frac{P(spam)}{P(cash=yes)}$<br>
$P(cash = yes \mid spam) = \frac{10}{30}$<br>
$P(spam) =  \frac{30}{100}$<br>
$P(cash = yes) = \frac{13}{100}$<br>
 = $\frac{10}{30} \times \frac{\frac{30}{100}}{\frac{13}{100}} = 0.769$ 

```
Exercise:
1) What is the probability of a ham email given that the word CASH! does not exist?
```

----

## Naive Bayes
# Why Naive?
<space>


```
      cash_yes cash_no furniture_yes furniture_no total
spam        10      20             6           24    30
ham          3      67            20           50    70
total       13      87            26           74   100
```


$P(spam \mid cash=yes \cap furniture=no) = \frac{P(cash=yes \cap furniture=no \mid spam) \times P(spam)}{P(cash=yes \cap furniture=no)}$

----

## Naive Bayes
# Why Naive?
<space>


```
      cash_yes cash_no furniture_yes furniture_no total
spam        10      20             6           24    30
ham          3      67            20           50    70
total       13      87            26           74   100
```


$P(spam \mid cash=yes \cap furniture=no) = \frac{P(cash=yes \cap furniture=no \mid spam) \times P(spam)}{P(cash=yes \cap furniture=no)}$

- Need $cash=yes \cap furniture=no$
  - $cash=yes \cap furniture=yes$
  - $cash=no \cap furniture=no$
  - $cash=no \cap furniture=yes$

----

## Naive Bayes
# Why Naive?
<space>

- As features increase, formula becomes very expensive
- Solution: assume each feature is independent of any other feature, given they are in the same class 
  - Independence formula: $P(A \cap B) = P(A) \times P(B)$
  - Called "class conditional independence"

----

## Naive Bayes
# Why Naive?
<space>

- As features increase, formula becomes very expensive
- Solution: assume each feature is independent of any other feature, given they are in the same class 
  - Independence formula: $P(A \cap B) = P(A) \times P(B)$
  - Called "class conditional independence":<br>
<br>
$P(spam \mid cash=yes \cap furniture=no) =$<br>
<br>

----

## Naive Bayes
# Why Naive?
<space>

- As features increase, formula becomes very expensive
- Solution: assume each feature is independent of any other feature, given they are in the same class 
  - Independence formula: $P(A \cap B) = P(A) \times P(B)$
  - Called "class conditional independence":<br>
<br>
$P(spam \mid cash=yes \cap furniture=no) =$<br>
<br>
$\frac{P(cash=yes \cap furniture=no \mid spam) \times P(spam)}{P(cash=yes \cap furniture=no)} =$<br> 
<br>

----

## Naive Bayes
# Why Naive?
<space>

- As features increase, formula becomes very expensive
- Solution: assume each feature is independent of any other feature, given they are in the same class 
  - Independence formula: $P(A \cap B) = P(A) \times P(B)$
  - Called "class conditional independence":<br>
<br>
$P(spam \mid cash=yes \cap furniture=no) =$<br>
<br>
$\frac{P(cash=yes \cap furniture=no \mid spam) \times P(spam)}{P(cash=yes \cap furniture=no)} =$<br> 
<br>
$\frac{P(cash=yes \mid spam) \times P(furniture=no \mid spam) \times P(spam)}{P(cash=yes) \times P(furniture=no)} =$<br>
<br>

----

## Naive Bayes
# Why Naive?
<space>

- As features increase, formula becomes very expensive
- Solution: assume each feature is independent of any other feature, given they are in the same class 
  - Independence formula: $P(A \cap B) = P(A) \times P(B)$
  - Called "class conditional independence":<br>
<br>
$P(spam \mid cash=yes \cap furniture=no) =$<br>
<br>
$\frac{P(cash=yes \cap furniture=no \mid spam) \times P(spam)}{P(cash=yes \cap furniture=no)} =$<br> 
<br>
$\frac{P(cash=yes \mid spam) \times P(furniture=no \mid spam) \times P(spam)}{P(cash=yes) \times P(furniture=no)} =$<br>
<br>
$\frac{\frac{10}{30} \times \frac{24}{30} \times \frac{30}{100}}{\frac{13}{100} \times \frac{74}{100}}$<br>
<br>

----

## Naive Bayes
# Why Naive?
<space>

- As features increase, formula becomes very expensive
- Solution: assume each feature is independent of any other feature, given they are in the same class 
  - Independence formula: $P(A \cap B) = P(A) \times P(B)$
  - Called "class conditional independence":<br>
<br>
$P(spam \mid cash=yes \cap furniture=no) =$<br>
<br>
$\frac{P(cash=yes \cap furniture=no \mid spam) \times P(spam)}{P(cash=yes \cap furniture=no)} =$<br> 
<br>
$\frac{P(cash=yes \mid spam) \times P(furniture=no \mid spam) \times P(spam)}{P(cash=yes) \times P(furniture=no)} =$<br>
<br>
$\frac{\frac{10}{30} \times \frac{24}{30} \times \frac{30}{100}}{\frac{13}{100} \times \frac{74}{100}}$<br>
<br>

```
Exercise:
1) What is the probability of a ham email given that the word CASH! exists and the word furniture does not?
```

----

## Naive Bayes
# The Laplace Estimator
<space>


```
      cash_yes cash_no furniture_yes furniture_no party_yes party_no total
spam        10      20             6           24         3       27    30
ham          3      67            20           50         0       70    70
total       13      87            26           74         3       97   100
```

<br>
<br>
$P(ham \mid cash=yes \cap party=yes) = \frac{P(cash=yes \mid ham) \times P(party=yes \mid ham) \times P(ham)}{P(cash=yes) \times P(party=yes)} = ?$

----

## Naive Bayes
# The Laplace Estimator
<space>


```
      cash_yes cash_no furniture_yes furniture_no party_yes party_no total
spam        10      20             6           24         3       27    30
ham          3      67            20           50         0       70    70
total       13      87            26           74         3       97   100
```

<br>
<br>
$P(ham \mid cash=yes \cap party=yes) = \frac{P(cash=yes \mid ham) \times P(party=yes \mid ham) \times P(ham)}{P(cash=yes) \times P(party=yes)} = \frac{\frac{3}{70} \times \frac{0}{70} \times \frac{70}{100}}{\frac{13}{100} \times \frac{3}{100}} = 0$

----

## Naive Bayes
# The Laplace Estimator
<space>


```
      cash_yes cash_no furniture_yes furniture_no party_yes party_no total
spam        10      20             6           24         3       27    30
ham          3      67            20           50         0       70    70
total       13      87            26           74         3       97   100
```

<br>
<br>
$P(ham \mid cash=yes \cap party=yes) = \frac{P(cash=yes \mid ham) \times P(party=yes \mid ham) \times P(ham)}{P(cash=yes) \times P(party=yes)} = \frac{\frac{3}{70} \times \frac{0}{70} \times \frac{70}{100}}{\frac{13}{100} \times \frac{3}{100}} = 0$

- To get around 0's, apply Laplace estimator
  - add 1 to every feature

----

## Naive Bayes
# Case Study: SMS spam filtering
<space>


```r
sms_data <- read.table("SMSSpamCollection.txt", stringsAsFactors = FALSE, sep = "\t", 
    quote = "", col.names = c("type", "text"))
str(sms_data)
```

```
'data.frame':	5574 obs. of  2 variables:
 $ type: chr  "ham" "ham" "spam" "ham" ...
 $ text: chr  "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..." "Ok lar... Joking wif u oni..." "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C"| __truncated__ "U dun say so early hor... U c already then say..." ...
```

```r
sms_data$type <- factor(sms_data$type)
```


----

## Naive Bayes
# Case Study: SMS spam filtering
<space>


```r
library(tm)
# create collection of text documents, a corpus
sms_corpus <- Corpus(VectorSource(sms_data$text))
sms_corpus
```

```
A corpus with 5574 text documents
```


----

## Naive Bayes
# Case Study: SMS spam filtering
<space>


```r
# look at first few text messages
for (i in 1:5) {
    print(sms_corpus[[i]])
}
```

```
Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
Ok lar... Joking wif u oni...
Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
U dun say so early hor... U c already then say...
Nah I don't think he goes to usf, he lives around here though
```


----

## Naive Bayes
# Case Study: SMS spam filtering
<space>


```r
# clean the data using helpful functions
corpus_clean <- tm_map(sms_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

# make each word in the corpus into it's own token each row is a message and
# each column is a word. Cells are frequency counts.
sms_dtm <- DocumentTermMatrix(corpus_clean)
```


----

## Naive Bayes
# Case Study: SMS spam filtering
<space>


```r
# create training and testing set
total_n <- nrow(sms_data)
train_ind <- sample(total_n, total_n * 2/3)

dtm_train_set <- sms_dtm[train_ind, ]
dtm_test_set <- sms_dtm[-train_ind, ]
corpus_train_set <- corpus_clean[train_ind]
corpus_test_set <- corpus_clean[-train_ind]
raw_train_set <- sms_data[train_ind, 1:2]
raw_test_set <- sms_data[-train_ind, 1:2]

# remove infrequent terms - not useful for classification
freq_terms <- c(findFreqTerms(dtm_train_set, 7))
corpus_train_set <- DocumentTermMatrix(corpus_train_set, list(dictionary = freq_terms))
corpus_test_set <- DocumentTermMatrix(corpus_test_set, list(dictionary = freq_terms))
```


----

## Naive Bayes
# Case Study: SMS spam filtering
<space>


```r
# convert frequency counts to 'yes' or 'no' implicitly weighing each term
# the same
convert <- function(x) {
    x <- ifelse(x > 0, 1, 0)
    x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
    return(x)
}

corpus_train_set <- apply(corpus_train_set, MARGIN = 2, FUN = convert)
corpus_test_set <- apply(corpus_test_set, MARGIN = 2, FUN = convert)
```


----

## Naive Bayes
# Case Study: SMS spam filtering
<space>


```r
library(e1071)

naive_model <- naiveBayes(x = corpus_train_set, y = raw_train_set$type)
predict_naive <- predict(naive_model, corpus_test_set)

naive_conf <- confusionMatrix(predict_naive, raw_test_set$type)$table
naive_conf
```

```
          Reference
Prediction  ham spam
      ham  1607   44
      spam    7  200
```


----

## Naive Bayes
# Case Study: SMS spam filtering
<space>


```r
library(e1071)

naive_model <- naiveBayes(x = corpus_train_set, y = raw_train_set$type)
predict_naive <- predict(naive_model, corpus_test_set)

naive_conf <- confusionMatrix(predict_naive, raw_test_set$type)$table
naive_conf
```

```
          Reference
Prediction  ham spam
      ham  1607   44
      spam    7  200
```


```
Exercise:
1) Calculate the true positive and false positive rate.  
2) Calculate the error rate (hint: error rate = 1 - accuracy)  
3) Set the Laplace = 1 and rerun the model and confustion matrix. Does this improve the model?
```

----

## Naive Bayes
# Summary
<space>

- Probabalistic approach
- Naive Bayes assumes features are independent, conditioned on being in the same class
- Useful for text classification
- Strengths
  - Simple, fast
  - Does well with noisy and missing data
  - Doesn't need large training set
- Weaknesses
  - Assumes all features are independent and equally important
  - Not well suited for numeric data sets


----

## Model Performance
# Measuring performance
<space>

- Classification
- Regression (more on this later)

----

## Model Performance
# Classification problems
<space>

- Accuracy is not enough
  - e.g. drug testing
  - class imbalance
- Best performance measure: Is classifier successful at intend purpose?

----

## Model Performance
# Classification problems
<space>

- 3 types of data used for measuring performance
  - actual values
  - predicted value
  - probability of prediction, i.e. confidence in prediction
- most R packages have a `predict()` function 
- confidence in predicted value matters
  - all else equal, choose the model that is more confident in its predictions
  - more confident + accuracy = better generalizer
  - set a paramter in `predict()` to `probability`, `prob`, `raw`, ...

----

## Model Performance
# Classification problems
<space>


```r
# estimate a probability for each class
confidence <- predict(naive_model, corpus_test_set, type = "raw")
as.data.frame(format(head(confidence), digits = 2, scientific = FALSE))
```

```
             ham           spam
1 0.969265077228 0.030734922772
2 0.000000000272 0.999999999728
3 0.999999987545 0.000000012455
4 0.000000000003 0.999999999997
5 0.999999274158 0.000000725842
6 0.997483049798 0.002516950202
```


----

## Model Performance
# Classification problems
<space>


```r
# estimate a probability for each class
spam_conf <- confidence[, 2]
comparison <- data.frame(predict = predict_naive, actual = raw_test_set[, 1], 
    prob_spam = spam_conf)
comparison[, 3] <- format(comparison[, 3], digits = 2, scientific = FALSE)
head(comparison)
```

```
  predict actual           prob_spam
1     ham    ham 0.03073492277184106
2    spam   spam 0.99999999972755882
3     ham    ham 0.00000001245462555
4    spam   spam 0.99999999999695421
5     ham    ham 0.00000072584174971
6     ham    ham 0.00251695020244934
```


----

## Model Performance
# Classification problems
<space>


```r
head(comparison[with(comparison, predict == actual), ])
```

```
  predict actual           prob_spam
1     ham    ham 0.03073492277184106
2    spam   spam 0.99999999972755882
3     ham    ham 0.00000001245462555
4    spam   spam 0.99999999999695421
5     ham    ham 0.00000072584174971
6     ham    ham 0.00251695020244934
```

```r
mean(as.numeric(comparison[with(comparison, predict == "spam"), ]$prob_spam))
```

```
[1] 0.9827
```


----

## Model Performance
# Classification problems
<space>


```r
head(comparison[with(comparison, predict != actual), ])
```

```
    predict actual           prob_spam
70      ham   spam 0.32261195298564260
142     ham   spam 0.22348642626700724
181     ham   spam 0.00074384650893715
226     ham   spam 0.00052525521655908
232     ham   spam 0.23899356386596546
248     ham   spam 0.00240698897952186
```

```r
mean(as.numeric(comparison[with(comparison, predict != "spam"), ]$prob_spam))
```

```
[1] 0.006759
```


----

## Model Performance
# Confusion Matrix
<space>

- Categorize predictions on whether they match actual values or not
- Can be more than two classes
- Count the number of predictions falling on and off the diagonals


```r
table(comparison$predict, comparison$actual)
```

```
      
        ham spam
  ham  1607   44
  spam    7  200
```


----

## Model Performance
# Confusion Matrix
<space>

- True Positive (TP)
- False Positive (FP)
- True Negative (TN)
- False Negative (FN)
- $Accuracy = \frac{TN + TP}{TN + TP + FN + FP}$
- $Error = 1 - Accuracy$

----

## Model Performance
# Kappa
<space>

- Adjusts the accuracy by the probability of getting a correct prediction by chance
- $k = \frac{P(A) - P(E)}{1 - P(E)}$
  - Poor < 0.2
  - Fair < 0.4
  - Moderate < 0.6
  - Good < 0.8
  - Excellent > 0.8

----

## Model Performance
# Kappa
<space>

- P(A) is the accuracy
- P(E) is the proportion of results where actual = predicted
  - $P(E) = P(E = class 1 ) + P(E = class 2)$
  - $P(E = class 1) = P(actual = class 1 \cap predicted = class 1)$
    - actual and predicted are independent so...

----

## Model Performance
# Kappa
<space>

- P(A) is the accuracy
- P(E) is the proportion of results where actual = predicted
  - $P(E) = P(E = class 1 ) + P(E = class 2)$
  - $P(E = class 1) = P(actual = class 1 \cap predicted = class 1)$
    - actual and predicted are independent so...
  - $P(E = class 1) = P(actual = class 1 ) \times P(predicted = class 1)$    
    - putting it all together... 

----

## Model Performance
# Kappa
<space>

- P(A) is the accuracy
- P(E) is the proportion of results where actual = predicted
  - $P(E) = P(E = class 1 ) + P(E = class 2)$
  - $P(E = class 1) = P(actual = class 1 \cap predicted = class 1)$
    - actual and predicted are independent so...
  - $P(E = class 1) = P(actual = class 1 ) \times P(predicted = class 1)$    
    - putting it all together...
  - $P(E) = P(actual = class 1) \times P(predicted = class 1) + P(actual = class 2) \times P(predicted = class 2)$

----

## Model Performance
# Kappa
<space>

- P(A) is the accuracy
- P(E) is the proportion of results where actual = predicted
  - $P(E) = P(E = class 1 ) + P(E = class 2)$
  - $P(E = class 1) = P(actual = class 1 \cap predicted = class 1)$
    - actual and predicted are independent so...
  - $P(E = class 1) = P(actual = class 1 ) \times P(predicted = class 1)$    
    - putting it all together...
  - $P(E) = P(actual = class 1) \times P(predicted = class 1) + P(actual = class 2) \times P(predicted = class 2)$
  
```
Exercise: 
1) Calculate the kappa statistic for the naive classifier.
```

----

## Model Performance
# Specificity and Sensitivity
<space>

- Sensitivity: proportion of positive examples that were correctly classified (True Positive Rate)
  - $sensitivity = \frac{TP}{TP + FN}$
- Specificity: proportion of negative examples correctly classified (True Negative Rate)
  - $specificity = \frac{TN}{FP + TN}$
- Balance aggressiveness and conservativeness
- Found in the confusion matrix
- Values range from 0 to 1

----

## Model Performance
# Precision and Recall
<space>

- Originally used in information retrieval
- Precision: proportion of positives that are truly positive
  - $precision = \frac{TP}{TP + FP}$
  - Precise model only predicts positive when it is sure. Very trustworthy model.
- Recall: proportion of true positives of all positives
  - $recall = \frac{TP}{TP + FN}$
  - High recall model will capture a large proportion of positives. Returns relevant results
- Easy to have high recall (cast a wide net) or high precision (low hanging fruit) but hard to have both high

----

## Model Performance
# Precision and Recall
<space>

- Originally used in information retrieval
- Precision: proportion of positives that are truly positive
  - $precision = \frac{TP}{TP + FP}$
  - Precise model only predicts positive when it is sure. Very trustworthy model.
- Recall: proportion of true positives of all positives
  - $recall = \frac{TP}{TP + FN}$
  - High recall model will capture a large proportion of positives. Returns relevant results
- Easy to have high recall (cast a wide net) or high precision (low hanging fruit) but hard to have both high

```
Exercise: 
1) Find the specificity, sensitivity, precision and recall for the Naive classifier.
```

----

## Model Performance
# F-score
<space>

- Also called the F1-score, combines both precision and recall into 1 measure
- $F_{1} = 2 \times \frac{precision \times recall}{precision + recall}$
- Assumes equal weight to precision and recall

----

## Model Performance
# F-score
<space>

- Also called the F1-score, combines both precision and recall into 1 measure
- $F_{1} = 2 \times \frac{precision \times recall}{precision + recall}$
- Assumes equal weight to precision and recall

```
Exercise: 
1) Calculate the F-score for the Naive classifier.
```

----

## Model Performance
# Visualizing performance: ROC
<space>

- ROC curves measure how well your classifier can discriminate between the positive and negative class
- As threshold increases, tradeoff between TPR (sensitivity) and FPR (1 - specificity)


```r
library(ROCR)

# create a prediction function
pred <- prediction(predictions = as.numeric(comparison$predict), labels = raw_test_set[, 
    1])

# create a performance function
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
```


----

## Model Performance
# Visualizing performance: ROC
<space>

- ROC curves measure how well your classifier can discriminate between the positive and negative class
- As threshold increases, tradeoff between TPR (sensitivity) and FPR (1 - specificity)

![plot of chunk roc_plot](figure/roc_plot.png) 


----

## Model Performance
# Visualizing performance: ROC
<space>

- The area under the ROC curve is the AUC
- Ranges from 0.5 (no predictive power) to 1.0 (perfect classifier)
    - 0.9 – 1.0 = outstanding
    - 0.8 – 0.9 = excellent 
    - 0.7 – 0.8 = acceptable
    - 0.6 – 0.7 = poor
    - 0.5 – 0.6 = no discrimination

----

## Model Performance
# Visualizing performance: ROC
<space>


```r
auc <- performance(pred, measure = "auc")
auc@y.values
```

```
[[1]]
[1] 0.9077
```


----

## Model Performance
# Holdout method
<space>

- In the kNN `Exercise`, we cheated...kind of

----

## Model Performance
# Holdout method
<space>

- In the kNN `Exercise`, we cheated...kind of
  - Train model - 50% of data
  - Tune parameters on validation set - 25% of data
      - (optionally) retrain final model on training and validation set (maximize data points)
  - Test final model - 25% of data


```r
new_data <- createDataPartition(sms_data$type, p = 0.1, list = FALSE)  # from caret package
table(sms_data[new_data, 1])
```

```

 ham spam 
 483   75 
```


----

## Model Performance
# Holdout method
<space>

- k-fold Cross Validation
   - Divide data into k random, equal sized partitions (k=10 is a commonly used)
   - Train the classifier on the K-1 parts
   - Test it on the Kth partition
   - Repeat for every K
   - Average the performance across all models - this is the Cross Validation Error
   - All examples eventually used for training and testing
   - Variance is reduced as k increases
   

```r
folds <- createFolds(sms_data$type, k = 10)  # from caret package
str(folds)
```

```
List of 10
 $ Fold01: int [1:558] 24 61 64 75 85 90 97 122 126 130 ...
 $ Fold02: int [1:557] 6 10 21 46 47 53 74 94 101 104 ...
 $ Fold03: int [1:557] 7 14 15 17 20 25 29 36 41 49 ...
 $ Fold04: int [1:558] 33 40 66 68 89 102 134 142 143 145 ...
 $ Fold05: int [1:556] 5 9 19 37 39 45 48 56 59 63 ...
 $ Fold06: int [1:558] 4 11 27 30 73 77 100 138 148 160 ...
 $ Fold07: int [1:558] 8 22 26 32 34 35 60 71 72 79 ...
 $ Fold08: int [1:557] 23 28 38 43 51 54 67 99 103 111 ...
 $ Fold09: int [1:558] 12 13 16 50 62 84 88 91 98 110 ...
 $ Fold10: int [1:557] 1 2 3 18 31 42 44 57 58 69 ...
```


----

## Regression
# Understanding Regression
<space>

- Predicting continuous value
- Concerned about relationship between independent and dependent variables
- Can be linear, non-linear, use decision trees, etc...
- Linear and non-linear regressions are called Generalized Minear Models

----

## Regression
# Linear regression
<space>

- $Y = \hat{\alpha} + \hat{\beta} X$
- $\hat{\alpha}$ and $\hat{\beta}$ are just estimates

![plot of chunk best_fit](figure/best_fit.png) 


----

## Regression
# Linear regression
<space>

- Distance between the line and each point is the error, or residual term
- Line of best fit: $Y = \alpha + \beta X + \epsilon$. &nbsp;&nbsp;&nbsp;Assumes:
  - $\epsilon$ ~ $N(0, \sigma^{2})$
  - Each point is IID (independent and identically distributed)
  - $\alpha$ is the intercept
  - $\beta$ is the coefficient
  - $X$ is the parameter
  - Both $\beta$ and $X$ are usually matrices

----

## Regression
# Linear regression
<space>

- Minimize $\epsilon$ by minimizing the mean squared error:
  - $MSE = \frac{1}{n} \sum_{i=1}^{n}\epsilon_{i}^{2} = \frac{1}{n} \sum_{i=1}^{n}(\hat{y_{i}} - y_{i})^{2}$
  - $y_{i}$ is the true/observed value
  - $\hat{y_{i}}$ is the approximation to/prediction of the true $y_{i}$
- Minimization of MSE yields an unbiased estimator with the least variance
- 2 common ways to minimize MSE:
  - analytical, closed form solution (e.g. `lm()` function does this)
  - approximation (e.g. gradient descent)

----

## Regression
# Gradient descent
<space>

- In Machine Learning, regression equation is called the hypothesis function
  - Linear hypothesis function: $h_{\theta}(x) = \theta_{0} + \theta_{1}x$
  - $\theta$ is $\{\beta_{0}, \beta{1}\}$, where $\beta_{0}$ is $\alpha$

----

## Regression
# Gradient descent
<space>

- In Machine Learning, regression equation is called the hypothesis function
  - Linear hypothesis function: $h_{\theta}(x) = \theta_{0} + \theta_{1}x$
  - $\theta$ is $\{\beta_{0}, \beta{1}\}$, where $\beta_{0}$ is $\alpha$
  - $X = \{x_{0}, x_{1}\}$
  - Make $x_{0}$ = 1
  - $h_{\theta}(x) = \beta \times t(X)$

----

## Regression
# Gradient descent
<space>

- In Machine Learning, regression equation is called the hypothesis function
  - Linear hypothesis function: $h_{\theta}(x) = \theta_{0} + \theta_{1}x$
  - $\theta$ is $\{\beta_{0}, \beta{1}\}$, where $\beta_{0}$ is $\alpha$
  - $X = \{x_{0}, x_{1}\}$
  - Make $x_{0}$ = 1
  - $h_{\theta}(x) = \beta \times t(X)$
- Goal remains the same: minimize MSE
  - define a cost (aka objective) function, $J(\theta_{0},\theta_{1})$
  - $J(\theta_{0},\theta_{1}) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x_{i}) - y_{i})^2$
  - $m$ is the number of examples

----

## Regression
# Gradient descent
<space>

- In Machine Learning, regression equation is called the hypothesis function
  - Linear hypothesis function: $h_{\theta}(x) = \theta_{0} + \theta_{1}x$
  - $\theta$ is $\{\beta_{0}, \beta{1}\}$, where $\beta_{0}$ is $\alpha$
  - $X = \{x_{0}, x_{1}\}$  
  - Make $x_{0}$ = 1
  - $h_{\theta}(x) = \beta \times t(X)$
- Goal remains the same: minimize MSE
  - define a cost (aka objective) function, $J(\theta_{0},\theta_{1})$
  - $J(\theta_{0},\theta_{1}) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x_{i}) - y_{i})^2$
  - $m$ is the number of examples
- Find a value for theta that minimizes $J$

----

## Regression
# Gradient descent
<space>

- Given a starting value, take a step along the slope
- Continue taking a step until minimum is reached

![plot of chunk grad_power](figure/grad_power.png) 


----

## Regression
# Gradient descent
<space>

- Given a starting value, take a step along the slope
- Continue taking a step until minimum is reached

![plot of chunk grad_power_2](figure/grad_power_2.png) 


----

## Regression
# Gradient descent
<space>

- Given a starting value, take a step along the slope
- Continue taking a step until minimum is reached

![plot of chunk grad_power_3](figure/grad_power_3.png) 


----

## Regression
# Gradient descent
<space>

- Given a starting value, take a step along the slope
- Continue taking a step until minimum is reached

![plot of chunk grad_power_4](figure/grad_power_4.png) 


----

## Regression
# Gradient descent
<space>

- Start with a point (guess)
- Repeat {
  - Determine a descent direction 
  - Choose a step size
  - Update
  }
- Until minimum is reached (or stopping criteria)

----

## Regression
# Gradient descent
<space>

- Start with a point (guess) &nbsp;&nbsp; &nbsp;  $x$
- Repeat {
  - Determine a descent direction &nbsp;&nbsp;&nbsp;  $-f^\prime$
  - Choose a step size &nbsp;&nbsp;&nbsp;  $\alpha$ (not the intercept!)
  - Update  &nbsp;&nbsp; &nbsp; $x:=x - \alpha f^\prime$
  }
- Until minimum is reached (or stopping criteria) &nbsp;&nbsp; &nbsp; $f^\prime ~ 0$

----

## Regression
# Gradient descent
<space>

- Update the value of $\theta$ by subtracting the first derivative of the cost function
- $\theta_{j}$ := $\theta_{j} - \alpha \frac{\partial}{\partial \theta_{j}}J(\theta_{0},\theta_{1})$
  - $j = 1, ..., p$ &nbsp;&nbsp; is the number of coefficients, or features
  - $\alpha$ is the step
  - $\frac{\partial}{\partial \theta_{j}}J(\theta_{0},\theta_{1})$ is the gradient
- Repeat until $J(\theta)$ is minimized

----

## Regression
# Gradient descent
<space>

- Using some calculus, we can show that
<br>
- $\frac{\partial}{\partial \theta_{j}}J(\theta_{0},\theta_{1})$
$=\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^{i}) - y^{i})(x^{i}_{j})$

----

## Regression
# Gradient descent
<space>

- And gradient descent formula becomes:
<br>
- $\theta_{j}$ := $\theta_{j} - \alpha\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^{i}) - y^{i})(x_{j}^{i})^{2}$
<br>
- Repeat until the cost function is minimized

----

## Regression
# Gradient descent
<space>

- Things to note
  - Choose the learning rate, alpha
  - Choose the stopping point
  - Local vs. global minimum
<br>
- Let's see it in action

----

## Regression example
# Gradient descent
<space>

![plot of chunk grad_ex_plot](figure/grad_ex_plot.png) 


----

## Regression example
# Gradient descent
<space>


```r
x <- cbind(1, x)  #Add ones to x  
theta <- c(0, 0)  # initalize theta vector 
m <- nrow(x)  # Number of the observations 
grad_cost <- function(X, y, theta) return(sum(((X %*% theta) - y)^2))
```


----

## Regression example
# Gradient descent
<space>


```r
gradDescent <- function(X, y, theta, iterations, alpha) {
    m <- length(y)
    grad <- rep(0, length(theta))
    cost.df <- data.frame(cost = 0, theta = 0)
    
    for (i in 1:iterations) {
        h <- X %*% theta  # Linear hypothesis function
        grad <- (t(X) %*% (h - y))/m
        theta <- theta - alpha * grad  # Update theta
        cost.df <- rbind(cost.df, c(grad_cost(X, y, theta), theta))
    }
    
    return(list(theta, cost.df))
}
```


----

## Regression example
# Gradient descent
<space>


```r
## initialize X, y and theta
X1 <- matrix(ncol = 1, nrow = nrow(df), cbind(1, df$X))
Y1 <- matrix(ncol = 1, nrow = nrow(df), df$Y)

init_theta <- as.matrix(c(0))
grad_cost(X1, Y1, init_theta)
```

```
[1] 5282
```

```r

iterations = 10000
alpha = 0.1
results <- gradDescent(X1, Y1, init_theta, iterations, alpha)
```


----

## Regression example
# Gradient descent
<space>

![plot of chunk grad_curve](figure/grad_curve.png) 


----

## Regression example
# Gradient descent
<space>


```r
grad_cost(X1, Y1, theta[[1]])
```

```
[1] 339.5
```

```r
## Make some predictions
intercept <- df[df$X == 0, ]$Y
pred <- function(x) return(intercept + x %*% theta)
new_points <- c(0.1, 0.5, 0.8, 1.1)
new_preds <- data.frame(X = new_points, Y = sapply(new_points, pred))
```


----

## Regression example
# Gradient descent
<space>


```r
ggplot(data = df, aes(x = X, y = Y)) + geom_point(size = 2)
```

![plot of chunk new_point](figure/new_point.png) 


----

## Regression example
# Gradient descent
<space>


```r
ggplot(data = df, aes(x = X, y = Y)) + geom_point() + geom_point(data = new_preds, 
    aes(x = X, y = Y, color = "red"), size = 3) + scale_colour_discrete(guide = FALSE)
```

![plot of chunk new_point_2](figure/new_point_2.png) 


----

## Regression example
# Gradient descent - Summary
<space>

- Minimization algorithm
- Approximation, non-closed form solution
- Good for large number of examples
- Hard to select the right $\alpha$
- Finds local not global minimum
- Traditional looping is slow - optimization algorithms are used in practice

----

## CV Error
# How many parameters are too many?
<space>

![plot of chunk multi_plot](figure/multi_plot.png) 


----

## CV Error
# How many parameters are too many?
<space>


```r
ggplot(df, aes(x = X, y = Y)) + geom_point() + geom_smooth(method = "lm", se = FALSE)
```

![plot of chunk multi_plot_2](figure/multi_plot_2.png) 

```r
summary(lm(Y ~ X, df))$adj.r.squared  # low R^2 - simple linear model won't fit
```

```
[1] 0.0529
```


----

## CV Error
# How many parameters are too many?
<space>

- Let's add some features


```r
df <- transform(df, X2 = X^2, X3 = X^3)
summary(lm(Y ~ X + X2 + X3, df))$coef[, 1]
```

```
(Intercept)           X          X2          X3 
     0.3363      6.5690    -26.9982     22.0071 
```

```r
summary(lm(Y ~ X + X2 + X3, df))$adj.r.squared
```

```
[1] 0.8082
```


----

## CV Error
# How many parameters are too many?
<space>

- Let's add even more features


```r
df <- transform(df, X4 = X^4, X5 = X^5, X6 = X^6, X7 = X^7, X8 = X^8, X9 = X^9, 
    X10 = X^10, X11 = X^11, X12 = X^12, X13 = X^13, X14 = X^14, X15 = X^15, 
    X16 = X^16, X17 = X^17, X18 = X^18, X19 = X^19, X20 = X^20)
line.fit <- lm(Y ~ X + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + X11 + X12 + 
    X13 + X14 + X15 + X16 + X17 + X18 + X19 + X20, df)
head(summary(line.fit)$coef[, 1])
```

```
(Intercept)           X          X2          X3          X4          X5 
  8.297e-03   1.842e+01  -7.101e+02   1.986e+04  -3.064e+05   2.903e+06 
```

```r
sqrt(mean((predict(line.fit) - df$Y)^2))  # Root MSE
```

```
[1] 0.08925
```


----

## CV Error
# How many parameters are too many?
<space>

- Use orthogonal polynomials to avoid correlated features
- `poly()` function


```r
ortho.coefs <- with(df, cor(poly(X, degree = 3)))
sum(ortho.coefs[upper.tri(ortho.coefs)])  # polynomials are uncorrelated
```

```
[1] -4.807e-17
```

```r
linear.fit <- lm(Y ~ poly(X, degree = 20), df)
summary(linear.fit)$adj.r.squared
```

```
[1] 0.9828
```

```r
sqrt(mean((predict(linear.fit) - df$Y)^2))
```

```
[1] 0.08859
```


----

## CV Error
# How many parameters are too many?
<space>

- When to stop adding othogonal features?

![plot of chunk polt_2](figure/polt_2.png) 


----

## CV Error
# How many parameters are too many?
<space>

- Use cross-validation to determine best degree


```r
x <- seq(0, 1, by = 0.005)
y <- sin(3 * pi * x) + rnorm(length(x), 0, 0.1)

indices <- sort(sample(length(x), round(0.5 * length(x))))

training.x <- x[indices]
training.y <- y[indices]

test.x <- x[-indices]
test.y <- y[-indices]

training.df <- data.frame(X = training.x, Y = training.y)
test.df <- data.frame(X = test.x, Y = test.y)

rmse <- function(y, h) return(sqrt(mean((y - h)^2)))
```


----

## CV Error
# How many parameters are too many?
<space>


```r
performance <- data.frame()
degrees <- 1:20
for (d in degrees) {
    fits <- lm(Y ~ poly(X, degree = d), data = training.df)
    performance <- rbind(performance, data.frame(Degree = d, Data = "Training", 
        RMSE = rmse(training.y, predict(fits))))
    performance <- rbind(performance, data.frame(Degree = d, Data = "Test", 
        RMSE = rmse(test.y, predict(fits, newdata = test.df))))
}
```


----

## CV Error
# How many parameters are too many?
<space>

![plot of chunk learning_plot](figure/learning_plot.png) 


----

## Regression
# Summary
<space>

- Minimize MSE of target function
- Analytically vs. approximation
- Gradient descent preferrable when lots of examples
- Use Cross Validation plot to determine optimal number of parameters

----

## Summary
# 
<space>

- Machine learning overview and concepts
- Visualizing Data
- kNN algorithm
- Naive Bayes
  - Probability concepts
  - Mobile Spam case study
- Model performance measures
- Regression

----

## Next Time
# 
<space>

- Logistic regression
- Decision Trees
- Clustering
- Dimensionality reduction (PCA, ICA)
- Regularization

----

## Resources
<space>

- [Machine Learning with R](http://www.packtpub.com/machine-learning-with-r/book)
- [Machine Learning for Hackers](http://shop.oreilly.com/product/0636920018483.do)
- [Elements of Statistical Learning](http://web.stanford.edu/~hastie/local.ftp/Springer/OLD/ESLII_print4.pdf)

----
