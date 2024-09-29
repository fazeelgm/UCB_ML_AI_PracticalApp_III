# Comparing Classifiers: Marketing Campaign Case Study

<p align='right'>
Practical Application III<br>
UC Berkeley ML/AI Professional Certification coursework<br>
Fazeel Mufti
</p>
  
**Resources**

* Moro, S., Rita, P., & Cortez, P. (2014). [Bank Marketing Dataset](https://doi.org/10.24432/C5K306). UCI Machine Learning Repository
  * [Accompanying Paper](CRISP-DM-BANK.pdf)

* `data/bank-additional-full.csv`: Full dataset for 41,188 campaign calls
* `data/bank-additional.csv`: Randomly sampled partial dataset with 4,118 campaign calls
* `MarketingCampaignCaseStudy.ipynb`: Jupyter notebook containing the Marketing Campaign Case Study for Classifier Comparisons

## Context
Our goal is to compare different Classification algorithms to predict if a client will subscribe to an offer made during a marketing campaign 
using a public dataset from the UCI Machine Learning repository [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing). 

The data is from a Portugese banking institution and is a collection of results from 17 Direct Marketing (DM) phone 
campaigns conducted by a Portugese Bank (Customer) between 
May 2008 and November 2010, corressponding to 79,354 contacts, who were offered attractive, long-term deposit applications. 
We will make use of the information provided by the authors in their paper accompanying the dataset [here](CRISP-DM-BANK.pdf) on how
they improved the dataset and features that were important in their model training. 

We will use the following **Methodlogy**:

* Conduct Exploratory Data Analysis (EDA) and develop a domain understanding of the attributes and feature distributions for suitability to data modeling
* Build baseline and default models using the following Classifiers
  * `LogisticRegression`
  * `KneighborsClassifier`
  * `DescisionTreeClassifier`
  * `SVC`: Support Vector Machine Classification
* Optimize the models by tuning relevant hyperparameters
* Compare and contrast tuned models based on their prediction abilities
* Recommendations based on our learnings to improve future DM camapaigns

**Business Objective**

Our business objective is to help the Customer optimize their DM campaigns in the future by predicting the likelihood of the 
campaign offer being accepted based on this data. We will now explore this data to develop an understanding of it's characteristics
so that we can generate machine learning (ML) models to help the Customer optimizetheir future DM campaigns and improve the 
likelihood of the campaign offer being accepted. 

## The Data

For each campaign contact across multiple attempts, various demographic and bank relationship attributes are provided. A separate column `y` has 
been provided showing whether the offer was accepted or not, i.e. was the capaign successful or not. 

**Client Attributes**
1. age (numeric)
1. job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
1. marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
1. education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
1. default: has credit in default? (categorical: 'no','yes','unknown')
1. housing: has housing loan? (categorical: 'no','yes','unknown')
1. loan: has personal loan? (categorical: 'no','yes','unknown')

**Last Contact Attributes**
8. contact: contact communication type (categorical: 'cellular','telephone')
1. month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
1. day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
1. duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

**Other Attributes**
12.campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
1. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
1. previous: number of contacts performed before this campaign and for this client (numeric)
1. poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

**Social and Cconomic Attributes**
16. emp.var.rate: employment variation rate - quarterly indicator (numeric)
1. cons.price.idx: consumer price index - monthly indicator (numeric)
1. cons.conf.idx: consumer confidence index - monthly indicator (numeric)
1. euribor3m: euribor 3 month rate - daily indicator (numeric)
1. nr.employed: number of employees - quarterly indicator (numeric)

**Target Variable**
21. y - has the client subscribed a term deposit? (binary: 'yes','no')


## Expoloratory Data Analysis

