# Palmer Penguins Species Classifier

An ML pipeline for the AIPI 510 Week 12 Interactive. It predicts penguin species from bill length, bill depth, flipper length, and body mass.

## Data and model

The script loads the [Palmer Penguins CSV](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv) and drops rows missing a selected measurement or species. It uses an 80/20 stratified train/test split with a random seed of 42. The pipeline fits a standard scaler and a logistic regression model on the training set.

## Result

Test accuracy was **100.0% (69 out of 69 penguins)** on the held-out split.
