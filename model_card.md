# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Model created as part of Udacity ML DevOps Engineering course

## Intended Use

The model used in this project aims to predict the salary of a person based on numeric (e.g., age, capital-gain etc.) as well as categoric features (workclass, education etc.)


## Training Data

The train/test split was done using a **split of 80/20** where data was taken from the census.csv file located in the ```data``` folder

## Evaluation Data

Based on the above split, we have 20% of the data left for testing.

## Metrics

Using a **Random Forest classifier** on the given dataset we achieved:


| Metric  | Value  |
|---|---|
|Fbeta| 0.9573210768220617  |
|Precision|  0.9298469387755102  |
|Recall|0.9433840181171141    | 


## Ethical Considerations

The model was not investigated in terms of bias as well as unbalanced dataset

## Caveats and Recommendations

When performing model performance testing on data slices it was observed that for some slices the model seems to be biased. 
Need to further investigated this. 