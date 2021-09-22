# LumiereProject

Research project using applied computer & data science. 

This package uses Keras as framework which provides a high-level neural networks API developed with a focus on enabling fast experimentation.

<p align="center">
  <img src="https://github.com/yiqiao-yin/YinsKerasNN/blob/master/pics/NNTraining.gif">
</p>

## Information

This repo contains the code used in the research project titled "A For-profit Model of Microcredit: Can Profit-driven Firms Improve Financial Inclusion in India?"

## Data
Data is sourced from the MIX Markets database of Microfinance Institutions (MFIs); MFIs in India or economically comparable regions are included.
The following features are extracted from the database: Number of Loan Officers, Gross Loan Portfolio, Number of Active Borrowers, Number of Loans Outstanding, Short Term Delinquency, Long term Delinquency, Average Loan per Borrower, Cost per Borrower, Borrowers per Loan Officer, Profit. 7 additional engineered features are included.
All features are normalized. 

## Baseline Model (Benchmark)
A Baseline Logistic Classifier to predict if a firm will be in profit given the list of features is first trained. Mean accuracy across 10 random states: 64.74%

## Proposed Model / Algorithm

<p>The logistic classifier classified only ~65% of the test-set correctly, which is not good enough for analysis. As such, two alternative models are proposed:</p>
<p>(1) A Decision Tree Classifier</p>
<p>(2) A Gradient-Boosted Tree Classifier</p>


### Algorithm

<p>Firms are classified into Profit (1) and Loss (0). Firms in loss were fewer in number, and so were randomly resampled to obtain balanced classes. A random train-test split with test weight 25% is carried out for 10 random states.</p>

<p>The Decision Tree Classifier is trained using an ID3 algorithm (based on information gain). Gini Importance is used to estimate the importance of each feature in determining profit.</p>

<p>The Gradient-Boosted Tree Classifier is also trained using an ID3 algorithm, but the model is retrained repeatedly to improve upon the errors of the previous iteration. Shapley values are used as measures of feature importance.</p> 

### Results
<p>
Mean test-set accuracy across 10 random states:
</p>
<p>
(1) Decision Tree Classifier: 87.94%
</p>
<p>
(2) Gradient-Boosted Tree Classifier: 88.78%
</p>

<p>Mean Feature Importance for non-engineered features (1) across 10 random states:</p>
<p>Number of Officers	0.026</p>
<p>Gross Loan Portfolio	0.094</p>
<p>Number of Active borrowers	0.086</p>
<p>Number of Loans Outstanding	0.118</p>
<p>Short Term Delinquency	0.076</p>
<p>Long Term Delinquency	0.056</p>
<p>Average Loan Per Borrower	0.068</p>
<p>Cost per Borrower	0.117</p>
<p>Borrowers per Loan Officer	0.052</p>

<p>Mean Shapley Importance for non-engineered features (2) across 10 random states:</p>
<p>Number of Officers	0.373</p>
<p>Gross Loan Portfolio	0.421</p>
<p>Number of Active borrowers	0.103</p>
<p>Number of Loans Outstanding	0.162</p>
<p>Short Term Delinquency	0.564</p>
<p>Long Term Delinquency	0.520</p>
<p>Average Loan Per Borrower	0.338</p>
<p>Cost per Borrower	0.920</p>
<p>Borrowers per Loan Officer	0.124</p>



