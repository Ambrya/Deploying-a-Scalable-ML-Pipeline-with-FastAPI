# Model Card

## Model Details

The model is a Random Forest classifier built with scikit-learn. It uses Census demographic and employment data to predict whether an individual earns more than $50,000 annually. The model was trained with a fixed random state of 42 to support reproducible results.

## Intended Use

This model is intended for educational purposes and demonstrates how to build, evaluate, and deploy a classification model through a FastAPI application. It should not be used to make employment, lending, insurance, or other high-impact decisions.

## Training Data

The model was trained using 80% of the Census Income dataset. The dataset includes demographic and employment-related features such as age, education, occupation, marital status, hours worked per week, and work class. Categorical features were transformed using one-hot encoding.

## Evaluation Data

The remaining 20% of the Census Income dataset was used for evaluation. A stratified train-test split preserved the distribution of the salary classes in both datasets. The evaluation data was processed using the encoder fitted to the training data.

## Metrics

The model was evaluated using precision, recall, and the F1 score. On the test dataset, the model achieved a precision of 0.7353, recall of 0.6378, and F1 score of 0.6831. Performance was also evaluated across every unique value of each categorical feature, with the results recorded in `slice_output.txt`.

## Ethical Considerations

The dataset contains sensitive demographic characteristics, including age, race, and sex. Historical and social biases in the data may influence the model’s predictions and produce unequal performance across demographic groups. Predictions should not be interpreted as judgments of an individual’s ability, worth, or future income potential.

## Caveats and Recommendations

The model is limited by the age, quality, and representativeness of the Census data. Some records contain unknown values, and the model may not generalize to current populations or individuals outside the dataset. Slice performance should be reviewed for disparities before any practical use. Future improvements could include hyperparameter tuning, fairness analysis, and evaluation with newer data.