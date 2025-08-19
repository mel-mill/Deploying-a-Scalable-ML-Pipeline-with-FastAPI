# Model Card: Income Classification with Random Forest


## Model Details

This model is a Random Forest Classifier trained to predict whether a person earns more than $50,000 per year. The model uses features from the U.S. Census dataset, including education, workclass, occumpation, and relationship status. It was built using scikit-learn in Python as part of Udacity MLOps course project.

## Intended Use

The model is intended for educational use only. It is part of a project to learn how to build, evaluate, and deploy machine learning models using a production-ready pipeline. It is not intended to be used in real-world applications.

## Training Data

The model was trained on the U.S. Census Income dataset provided in the course. The data includes both categorical and numerical features. One-hot encoding was used for categorical variables, and the salary label was converted to binary format (<=50K or >50K). The training data was 80% of the full dataset.

## Evaluation Data

The evaluation data was the remaining 20% of the dataset, separated using `train_test_split` with a fixed random seed (random_state=42) to ensure reproducibility.

## Metrics

The model was evaluated using precision, recall, and F1 score on the test set.

- **Precision**: 0.7419
- **Recall**: 0.6384
- **F1 Score**: 0.6863

These metrics were computed using scikit-learn's classification metrics.

## Ethical Considerations

The model was trained only for practice and has not been reviewed for fairness or bias. Since it uses demographic data, it may reflect or amplify existing societal biases in the original dataset.

## Caveats and Recommendations

This model was created as a learning exercise and should not be used to make decisions in real-world settings.
