# Most Likely To Quit: Three Machine Language Approaches

This project is based on a hypothetical dataset on HR Employee attrition and performance. It has various data describing employee characteristics. Machine Learning can help to understand and determine how these factors relate to workforce attrition.

This project shows that data analytics can be of use to HR personnel in identifing which employees are most likely to quit. This information can be passed on to business unit managers to interdict and address the issues affecting employee satisfaction. Additionally, senior management can use the anonymized results to examine if there are systemic issues – e.g., an over-reliance on overtime to complete projects, possible lack of mentorship for recent hires, inadequate compensation for sales staff– that can be addressed with changes in organizational philosophy or programs. 

It should come as no surprise that several of the key variables suggested by the models are often those which are cited in literature as significantly contributory towards voluntary termination (Griffeth, R. A Meta-Analysis of Antecedents and Correlates of Employee Turnover. 2000. ) However, these models show that data analytics can provide a finer level of granularity in identifying which employees are most at risk, enabling HR staff to preemptively intercede before someone decides to make a change, thus saving the organization the expenses associated with replacing key personnel.   

# Approach

Three different algorithms are used in this analysis: logistic regression, random forests, and support vector classification. 
All three models are robust, appropriate for the data and provide results that can be reasonably interpreted. Default settings were initially used for each model run, then tuned according to each model's parameters. 

The logistic regression model uses an inverse regularization parameter, defined as C = 1/(Lamda), which is a penalty term used to regulate overfitting. The random forest classifier is tuned using a list of number of estimators, from 100 to 500 in 50 step increments. Both models use a five-fold stratified cross-validation function to mitigate against class imbalances. Support vector machine takes a tuning parameter called gamma, which is the inverse of the radius of influence of samples selected by the model as support vectors. It is defined as 1 / (n_features * X.var( )). 

# Results

Since the AUC is a better evaluation metric than an error matrix for this type of classification problem (Agresti), the ROC curves measure each model's efficacy versus the holdout (testing) dataset.  While all three models scored relatively similarly (Fig.9), the logistic regression model (0.822) scored higher than either the random forest model (0.783) or support vector classification model (0.767). Similarly, it has the highest precision, recall and f1 score, and the smallest logloss score. From this we can conclude that the logistic regression model is best suited to interpret this data. Finally, a sorted printout of the exponent of the log-odds ratios of features (Table 1) helps determine the most important factors correlated to attrition. There are many that increase odds of quitting above 50%, but five have measures greater than 200%. These are: JobInvolvement_1 (410%), OverTime_Yes (354%), JobLevel_1 (338%), BusinessTravel_Travel_Frequently (240%) and JobSatisfaction_1 (213%). 

# Conclusions
  
From this data we can divine the profile of a "typical" quit employee: a low-level employee working overtime and frequently traveling for business who is very unsatisfied with their position. Additionally, the data suggests this person might be single and work in sales or possess a technical degree.
