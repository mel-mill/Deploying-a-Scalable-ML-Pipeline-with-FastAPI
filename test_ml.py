import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics

def test_data_split():
    """
    Test that train/test split adds up to full dataset.
    """
    data = pd.read_csv("data/census.csv")
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    assert len(train) + len(test) == len(data)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)



def test_model_classifier():
    """
    Test that train_model returns a RandomForestClassifier
    """
    data = pd.read_csv("data/census.csv")
    train, _ = train_test_split(data, test_size=0.2, random_state=42)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]
    X_train, y_train, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)



def test_metrics_types():
    """
    Test that compute_model_metrics returns three floats.
    """
    data = pd.read_csv("data/census.csv")
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    model = train_model(X_train, y_train)
    preds = model.predict(X_test)

    precision, recall, f1 = compute_model_metrics(y_test, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)
