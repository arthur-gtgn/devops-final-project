import pytest
from mushroom_ml.train import train_model
from sklearn.ensemble import RandomForestClassifier


def test_train_model_default_params():
    # Test that train_model returns a RandomForestClassifier with default params
    model = train_model()
    assert isinstance(model, RandomForestClassifier)
    # Check that default parameters are set correctly via get_params()
    params = model.get_params()
    assert params['criterion'] == 'gini'
    assert params['n_estimators'] == 100
    assert params['max_depth'] is None
    assert params['bootstrap'] is True


def test_train_model_custom_params():
    # Test that custom parameters are applied correctly
    params = {
        'criterion': 'entropy',
        'n_estimators': 10,
        'max_depth': 5,
        'bootstrap': False
    }
    model = train_model(**params)
    assert isinstance(model, RandomForestClassifier)
    # Validate custom parameters via get_params()
    got = model.get_params()
    assert got['criterion'] == params['criterion']
    assert got['n_estimators'] == params['n_estimators']
    assert got['max_depth'] == params['max_depth']
    assert got['bootstrap'] == params['bootstrap']
