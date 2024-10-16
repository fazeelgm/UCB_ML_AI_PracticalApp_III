# Comparing Classifiers: Marketing Campaign Case Study

<p align='right'>
Practical Application III<br>
UC Berkeley ML/AI Professional Certification coursework<br>
Fazeel Mufti
</p>
  
**Resources**

* Moro, S., Rita, P., & Cortez, P. (2014). [Bank Marketing Dataset](https://doi.org/10.24432/C5K306). UCI Machine Learning Repository
  * [Accompanying Paper](misc/CRISP-DM-BANK.pdf)

* `data/bank-additional-full.csv`: Full dataset for 41,188 campaign calls
* `data/bank-additional.csv`: Randomly sampled partial dataset with 4,119 campaign calls
* `MarketingCampaignCaseStudy.ipynb`: Jupyter notebook containing the Marketing Campaign Case Study for Classifier Comparisons

## Context
Our goal is to compare different Classification algorithms to predict if a client will subscribe to an offer made during a marketing campaign 
using a public dataset from the UCI Machine Learning repository [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing). 

The data is from a Portugese banking institution and is a collection of results from 17 Direct Marketing (DM) phone 
campaigns conducted by a Portugese Bank (Customer) between 
May 2008 and November 2010, corressponding to 79,354 contacts, who were offered attractive, long-term deposit applications. 
We will make use of the information provided by the authors in their [paper accompanying the dataset](misc/CRISP-DM-BANK.pdf) on how
they improved the dataset and features that were important during their model training. 

**Methodology**

We will use the following methodology:

* Conduct Exploratory Data Analysis (EDA) and develop a domain understanding of the attributes and feature distributions for suitability to data modeling
* Build baseline and default models using the following standard classification models:
  * `LogisticRegression`
  * `KneighborsClassifier`
  * `DescisionTreeClassifier`
  * `SVC`: Support Vector Machine based classifier
* Optimize the models by tuning relevant hyperparameters
* Compare and contrast tuned models based on their prediction abilities
* Recommendations based on our learnings to improve future Customer Direct Marketing campaigns

**Business Objective**

Our business objective is to help the Customer optimize their future DM campaigns by predicting the likelihood of a 
campaign offer being accepted based on the current dataset. We will first explore this data to develop an understanding of it's characteristics, and 
then generate machine learning (ML) models to help the Customer optimizetheir future campaigns and improve the 
likelihood of the campaign offer being accepted. 

## The Data

For each campaign contact across multiple attempts, various demographic and bank relationship attributes are provided. A separate column `y` has 
been provided showing whether the offer was accepted or not, i.e. was the campaign successful or not. 

**Client Attributes**
1. `age` (numeric)
1. `job`: type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
1. `marital`: marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
1. `education`: (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
1. `default`: has credit in default? (categorical: 'no','yes','unknown')
1. `housing`: has housing loan? (categorical: 'no','yes','unknown')
1. `loan`: has personal loan? (categorical: 'no','yes','unknown')

**Last Contact Attributes**

8. `contact`: contact communication type (categorical: 'cellular','telephone')
1. `month`: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
1. `day_of_week`: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
1. `duration`: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

**Other Attributes**

12. `campaign`: number of contacts performed during this campaign and for this client (numeric, includes last contact)
1. `pdays`: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
1. `previous`: number of contacts performed before this campaign and for this client (numeric)
1. `poutcome`: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

**Social and Cconomic Attributes**

16. `emp.var.rate`: employment variation rate - quarterly indicator (numeric)
1. `cons.price.idx`: consumer price index - monthly indicator (numeric)
1. `cons.conf.idx`: consumer confidence index - monthly indicator (numeric)
1. `euribor3m`: euribor 3 month rate - daily indicator (numeric)
1. `nr.employed`: number of employees - quarterly indicator (numeric)

**Target Variable**

21. `y` - has the client subscribed a term deposit? (binary: 'yes','no')

## Expoloratory Data Analysis

> **Note**: Please refer to the [`MarketingCampaignCaseStudy.ipynb`](MarketingCampaignCaseStudy.ipynb) for detailed work that is being presented in summary format here.

### Data Analysis

1. Figure 1: Overall feature inspection
   * The data was relatively clean - No missing or null values were discovered
   * Overall distributions look reasonable - we can't exclude any features at this stage as it may be relevant for modeling. We can revisit this when we tune the model hyperparamters
   * Some data concerns that we can note for now and may have to come back to later:
     * `unknown` values for `marital`, `default`, `housing` and `loan` columns: We ignored them for now, but if these features are important during modeling, these missing values can be treated as a possible class label or using deletion or imputation techniques. This is specially concerning for `default` cases where the `unknown` counts are significant
     * `pdays` distribution shows that majority of the resepondents were not previously contacted (999), so this may be a red flag for removing this column
1. Figure 2: The target variable `y` was not much of a concern with respect to outliers since this is a binary classification problem. Only 11.3% of the campaigns were successful so the distribution is unbalanced and we will take this into account as we tune the models
2. Figure 3: Performed Correlation analysis after `OrdinalEncoder` was used to encode categorical features to see inter-feature relationships
   * Most of the data seems to have reasonable correlations to the target `y` variable with `duration` and `previous` values of last call with customer showing strong positive relation to success (`y` = `yes`)
   * It was deemed too early to eliminate any features till we get preliminary results from our models

<table style="width:100%"><tr>
  <td width="60%"><img src="images/feature_distributions.png" border="0"/><em>Figure 1: Feature Distributions</em></td>
  <td width="40%">
    <img src="images/pie-acceptance-overall.png" border="0"/><em>Figure 2: Target Distribution</em>
    <br><br><br>
    &nbsp;&nbsp;&nbsp;
    <img src="images/heatmap_cmatrix.png" border="0"/><em>Figure 3: Correlation Matrix</em>
  </td>
</tr></table>

### Feature Engineering

Next, we prepared the data for modeling:

1. The data was split into training and validation sets using an 80/20 split with stratification over the target `y` to ensure the training and validation data was representative of the overall distribution
2. From this point on, we use the same `random_state=42` for all data and modeling to reduce shuffling artifacts from run-to-run
3. The data was scaled using `StandardScaler` for the initial runs

## Model Development

### Baseline & Default Models

Before we build our first model, we want to establish a baseline. We used the Scikit ML Library provided `DummyClassifier` with `strategy='uniform'` to make _untrained_ predictions with equal probability among the two possible classes: Successful and Unsuccessful, without taking any of the campaign features into account. As expected, the predictions from the DummyClassifier (Figure 4) are about as good as a coin toss (50/50), and this is the baseline score for our business use case!

#### Evaluation Metrics

We next proceeded to benchmark the performance of the Logistic Regression, KNN algorithm, Decision Tree and SVM models by using the default settings of the models to fit and score each one using the following metrics:

1. Accuracy: Proportion of correctly predicted instances (both positive and negative) out of the total instances in the dataset
2. Precision: Proportion of positive predictions that are actually correct
3. Recall: Proportion of actual positives that are correctly identified by the model
4. F1-score: The harmonic mean of precision and recall, providing a single metric that balances both
5. ROC-AUC score: The performance of a classifier across different threshold values
6. Confusion Matrix: Summary of the prediction results by showing the counts of true positive (TP), true negative (TN), false positive (FP), and false negative (FN) classifications


<table style="width:100%"><tr>
  <td width="100%"><em>Figure 4: Default Model Results</em><img src="images/table_models_defaults.png" border="0"/></td>
</tr></table>

We can now compare the results from all our base Models! 

* All our models did much better than the 50% baseline of the `DummyClassifier`
* `LogisticRegression` performed the best with the highest Test Accuracy Score on the validation data, i.e. after being trained on the training data, how it did against the held-out test dataset in correctly predicting the offer acceptance (both positive and negative) across all calls (also shown by the highest AUC score). It was also the fastest to train and overall is a good candidate for using this as the final model
* `SVC` came in second, but took almost an order of magnitude longer to train with a marginal improvement over `LogisticRegression`. However, `SVC` scored the highest on the AUC score, showing a better balance among true positive and negative classification.
* Both `DecisionTreeClassifier` and `KNeighborsClassifier` were close in Test Accuracy and were faster than `LogisticRegression` but had lower AUC scores
* Overall, all four models were relatively comaparable with no standouts for elimination at this point

### Model Tuning

We will now try to improve the above results by:

* More feature engineering and data exploration
* Hyperparameter tuning to optimze the models
* Create an ML pipeline using `GridSearchCV` to optimize the hyperparameters of our models and identify the parameters that provide the best metrics for the training dataset
* We used 5-fold cross-validation for all models with the following model and param_grid definitions:
```
# Define the models to be tuned
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'SVC': SVC(probability=True, max_iter=10000)
}

# Define the hyperparameters to test
param_grid = {
    'LogisticRegression': {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
    },
    
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    
    'DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    
    'SVC': {
        'C': np.logspace(-3, 4, 5).round(4),
        'kernel': ['linear', 'poly', 'rbf'],
        'degree': [1, 3],    # For poly only - 5 didn't win
        'gamma': ['scale', 'auto'],
    }
}
```
It takes about 8 hours to complete the pipeline on the full dataset so the smaller dataset was used during hyperparameter tuning.

Optimizations performed during the tuning cycles:

* Switch to `MinMaxScaler` from `StandardScaler`
* Tuned `max_iter`, `C`, `kernel` and `degree` parameters based on results
* `LogisticRegression`: switched to `class_weight=balanced`

## Evaluation

### Key Findings

<table style="width:100%"><tr>
  <td width="100%"><em>Figure 5: Tuned (optimized) Model Results</em><img src="images/table_models_tuned.png" border="0"/></td>
</tr></table>

1. `SVC` and `DecisionTreeClassifier` performed better than the other models this time with a 91% accuracy, but `SVC` came in with a higher AUC score so it has a lit bit of an edge from a classfication perspective
2. However, `SVC` took drastically longer to train, so this is a concern for using this model.
1. `DecisionTreeClassifier` had the highest F1 score on an impbalanced `y` distribution, suggesting a good balance on both the Precision and Recall scores. It also had the fastest training time with reasonable scores on AUC and F1 and balanced the time vs score criteria well
1. `LogisticRegresssion` had the highest AUC score with high recall, classifying the most successfull campaigns correctly (835) and quickly
1. `KNeighborsClassifier` did reasonaly well with moderate training time and the higest precision score

### Model Interpretation

We now shift our focus to interpreting the results from comparisons and understand the weighting of our training features on campaign success. We analyzed each model in more detail on the basis of our evaluation metrics mentioned above:

<table style="width:100%"><tr>
  <td width="50%"><img src="images/confusion_matrices.png" border="0"/><em>Figure 6: Confusion Matrices</em></td>
  <td width="50%">
    <img src="images/roc_curves.png" border="0"/><em>Figure 7: ROC/AUC Curves</em>
    <br><br><br>
    &nbsp;&nbsp;&nbsp;
    <img src="images/feature_importances.png" border="0"/><em>Figure 8: Feature Importances</em>
  </td>
</tr></table>

As we have already discussed that the top models perfromed very similarly and this is shown in their correct classification results (confusion matrices). Their relative perfromances during tuning across various hyperparameter thresholds as shown by the AUC graphs, shows us that we were able to get the best parameters for each given our data.

Looking at the above Feature Importances from the two models that provide this information, `LogisticRegresssion` and `DecisionTreeClassifier`, we see that each tuned model gives different weight to individual features. Decision trees are easy to interpret and show how the model came to its _decision_ for individual samples. Our tuned `DecisionTreeClassifier` was able to achieve the highest 93.46% accuracy on the training data, with allocated `max_depth=10`, out of the tuned models (likely a little over-fitted). Due to its overall performance and best time to train, it is instructive to visually see how it came decide on it's decision matrix.

We investigated two different methods for plotting the resulting decision tree to understand the output, as it is a good way to explain the prediction path to the customer and this will help them design better campaigns in the future:

* SciKit-learn DecisionTreeClassifier [Link to full tree -> Click then Right Click to Open In New Window](images/decision_tree.png)
* Dtreeviz Library [Link to full tree -> Click then Right Click to Open In New Window](images/decision_dtreeviz.svg)

You can view the full trees from both tree plotting methods, but we liked the advanced organization features of the `dtreeviz` library. Here we show the partial decision tree from our optimized model, as well as the prediction path for the individual 4,058th sample from our dataset that the model used to classify it as a Success (`yes`):
```
0.29 <= age 
0.5 <= month 
day_of_week < 0.38
duration < 0.04
campaign < 0.03
pdays < 0.01
euribor3m < 0.02
nr.employed < 0.47
```

<table style="width:100%"><tr>
  <td width="100%"><em>Figure 9: Prediction path for the 4,058th sample</em><img src="images/decision_dtreeviz_row_4058_snippet.png" border="0"/></td>
</tr></table>

### Recommendations

Since we don't get the individual feature importance from all our models, we then calculated the Permutation Importance to measure the change in our model's performace when a feature value is randomly shuffled (10 iterations) to see how much the model relies on that feature for its predictions. This helps us determine collinearity between features not captured by the model cofficients and evaluate the impact of changing the feature on the model performance, and to make recommendations based on these learnings that will impact campaign design going forward.

<table style="width:100%"><tr>
  <td width="100%"><em>Figure 10: Permutation Importance</em><img src="images/permutation_importance.png" border="0"/></td>
</tr></table>

Based on these importances and what we learned from inidividual models, we can make the following recommendations to improve future campaigns:

1. There is a strong correlation between the call `duration` including previous call metrics (`pdays`, `month`, `contact`, and the lieklihood of acceptance of the offer. Therefore, we should design a campaign call script to enage the customer and extend the conversation as much as possible. This could be either by offering increasing rewards, understanding the reason they are not accepting the current offer or bringing in other value-add services to the discussions
2. The socio-economic attributes (`euribor3m`,`cons.price.idx`, `nr.employed`) positively impacting the predictions so there is a need to have this external data available when selecting the target customer segments who will be included in the campaigns
3. Since multiple models scored above 90% accuracy, we have high confidence of success if we can target the campaigns to similiar customers with around 11% conversion. The guidance provided here on segment-selection will help increase this conversion rate

Since our Case Study benefitted from the work of the original Research Team and came inline with their results, we still learned a lot about applying Classification models to real-world applications. The mehtodogy used gives us confidence that we can apply our learnings to other domains in the future.
