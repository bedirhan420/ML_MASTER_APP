# ML_MASTER_APP

### This project consists of 5 stages:

## 1) Data Preprocessing

![data_preprocessing](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/data_preprocessing.png?raw=true)

After uploading a CSV file, you can:

- View the first 5 rows
- View the columns
- View the shape
- View statistical information
- View and fill missing (null) values
- View unique values and their counts
- View data types
- View the correlation matrix
- Perform outlier removal
- In addition to these, you can also perform sorting, grouping, and filtering operations.

## 2) Feature Engineering

![feature_engineering](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/feature_engineering.png?raw=true)

On this page, the following feature engineering operations can be performed:

- Mathematical operations
- Logarithms
- Comparison with numbers
- Column comparison
- Date extraction
- Shift
- Modulus calculation
- As a result of these operations, new columns are added to the dataset.

## 3) Hypothesis Testing

On this page, the hypothesis testing automation provides an automated solution for analyzing datasets and performing statistical tests based on the properties of selected columns. It determines the appropriate test type based on the column data and executes the test automatically.

Supported Statistical Tests:

- Pearson Correlation Test
- Spearman Correlation Test
- Chi-Square Test
- Independent T-Test
- Mann-Whitney U Test
- ANOVA (Analysis of Variance)
- A/B Test


Features:

- Automated Test Selection: Determines the appropriate statistical test based on the types of selected columns (numerical or categorical) and their distributions.

- Normallity Check: Applies the Shapiro-Wilk test to check for normality of numerical data.

- Test Execution: Performs the identified test and returns the test statistic and p-value.

- Dynamic Columns: Users can select any two columns from the dataset for analysis.

Example Usages:

ANOVA:
![anova](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/anova.png?raw=true)

CHI SQUARE:
![ki_kare](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/ki_kare.png?raw=true)

## 4) Data Visualization

![data_visualization](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/data_visualization.png?raw=true)

On this page, you can visualize the data with the following types of charts:

1. **Bar Chart**
   - One Parameter
     - ![bar1](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/bar1.png?raw=true)
   - Two Parameters
     - ![bar2](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/bar2.png?raw=true)

2. **Horizontal Bar Chart**
   - One Parameter
     - ![horbar1](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/horbar1.png?raw=true)
   - Two Parameters
     - ![horbar2](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/horbar2.png?raw=true)

3. **Line Plot**
   - Without Hue
     - ![line1](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/line1.png?raw=true)
   - With Hue
     - ![line2](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/line2.png?raw=true)

5. **Box Plot**
   - One Parameter
     - ![box1](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/box1.png?raw=true)
   - Two Paramater Without Hue
     - ![box2](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/box2.png?raw=true)
   - Two Paramater With Hue
     - ![box3](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/box3.png?raw=true)

6. **Pie Chart**
   - ![pie](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/pie_chart.png?raw=true)

7. **Pairplot**
   - ![pairplot](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/pairplot.png?raw=true)

8. **Scatter Plot**
  - Without Hue
     - ![scatter1](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/scatter1.png?raw=true)
  - With Hue
     - ![scatter2](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/scatter2.png?raw=true)

9. **Histogram**
   - Without Hue
     - ![hist1](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/hist1.png?raw=true)
  - With Hue
     - ![hist2](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/hist2.png?raw=true)

10. **Heatmap**
   - ![heatmap](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/heatmap.png?raw=true)

## 5) Model Training

On this page, under three main categories, the following algorithms are available:

- **Regression:**
  - Linear Regression
  - Polynomial Regression
  - Logistic Regression
  - Support Vector Regression (SVR)
  - Decision Tree Regressor
  - Random Forest Regressor 
  - TabPFN Regressor

- **Classification:**
  - K-Nearest Neighbors (KNN) Classifier
  - Support Vector Classifier (SVC)
  - Naive Bayes Classifier
  - Decision Tree Classifier
  - Random Forest Classifier
  - XGBoost Classifier
  - CatBoost Classifier
  - Gradient Boosting Classifier
  - TabPFN Classifier

- **Clustering:**
  - K-Means Clustering
  - Agglomerative Clustering
  - DBSCAN
  - Dendrogram

### Example: XGBoost Algorithm on the Iris Dataset

1. Choose the classification model type and the `XGBClassifier` algorithm. Define the X and y columns and specify the test size.
   ![model1](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/model1.png?raw=true)

2. Click the "Fit" button and wait. After the model training is completed, view the analysis graphics of the model.
   ![model2](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/model3.png?raw=true)

3. When you close the chart tab, view the model metrics digitally and save the model if desired.
   ![model3](https://github.com/bedirhan420/ML_MASTER_APP/blob/main/IMAGES/model2.png?raw=true)
