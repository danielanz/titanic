"""
Model pipelines for Titanic classification.
Includes custom imputers and preprocessing components for different models.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


# =============================================================================
# Column Definitions
# =============================================================================

LR_NUMERIC_FEATURES = ['age', 'family_size', 'fare_per_person', 'ticket_frequency']
TREE_NUMERIC_FEATURES = ['age', 'family_size', 'fare_per_person', 'sibsp', 'parch', 'ticket_frequency']
BINARY_FEATURES = ['sex', 'has_cabin', 'is_female_or_child', 'is_alone']
CATEGORICAL_FEATURES = ['pclass', 'embarked', 'title', 'sex_pclass', 'deck']


# =============================================================================
# Custom Imputers
# =============================================================================

class GroupedMedianAgeImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing Age values using median grouped by Sex and Pclass.
    Falls back to global median if group not found.
    """
    
    def __init__(self):
        self.group_medians_ = None
        self.global_median_ = None
    
    def fit(self, X, y=None):
        df = self._to_dataframe(X)
        self.group_medians_ = df.groupby(['sex', 'pclass'])['age'].median().to_dict()
        self.global_median_ = df['age'].median()
        return self
    
    def transform(self, X):
        df = self._to_dataframe(X).copy()
        
        mask = df['age'].isnull()
        for idx in df[mask].index:
            sex = df.loc[idx, 'sex']
            pclass = df.loc[idx, 'pclass']
            group_key = (sex, pclass)
            
            median_val = self.group_medians_.get(group_key, self.global_median_)
            if pd.isnull(median_val):
                median_val = self.global_median_
            df.loc[idx, 'age'] = median_val
        
        return df
    
    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        raise ValueError("Input must be a pandas DataFrame")


# =============================================================================
# Logistic Regression Components
# - GroupedMedianAgeImputer for Age (first step)
# - StandardScaler for numerical
# - Binary to 0/1
# - OneHotEncoder for categorical
# =============================================================================

# Age imputer (applied first to full DataFrame)
lr_age_imputer = GroupedMedianAgeImputer()

# Numeric transformer: impute with median, then scale
lr_numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Binary transformer: impute with most_frequent, encode to 0/1
lr_binary_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

# Categorical transformer: impute with most_frequent, one-hot encode
lr_categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Logistic Regression column transformer
lr_column_transformer = ColumnTransformer(
    transformers=[
        ('num', lr_numeric_transformer, LR_NUMERIC_FEATURES),
        ('bin', lr_binary_transformer, BINARY_FEATURES),
        ('cat', lr_categorical_transformer, CATEGORICAL_FEATURES)
    ],
    remainder='drop'
)

# Logistic Regression classifier
lr_classifier = LogisticRegression(
    max_iter=1000,
    solver='saga',
    l1_ratio=0.5,  # Controls regularization: 0=L2, 1=L1, 0<x<1=ElasticNet
    random_state=42
)

# Logistic Regression pipeline (Age imputer -> ColumnTransformer -> Classifier)
lr_pipeline = Pipeline(steps=[
    ('age_imputer', lr_age_imputer),
    ('preprocessor', lr_column_transformer),
    ('classifier', lr_classifier)
])


# =============================================================================
# Random Forest Components
# - GroupedMedianAgeImputer for Age (first step)
# - No scaling for numerical
# - Binary to 0/1
# - OrdinalEncoder for categorical
# =============================================================================

# Age imputer (applied first to full DataFrame)
rf_age_imputer = GroupedMedianAgeImputer()

# Numeric transformer: impute with median, no scaling
rf_numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Binary transformer: impute with most_frequent, encode to 0/1
rf_binary_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

# Categorical transformer: impute with most_frequent, ordinal encode
rf_categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Random Forest column transformer
rf_column_transformer = ColumnTransformer(
    transformers=[
        ('num', rf_numeric_transformer, TREE_NUMERIC_FEATURES),
        ('bin', rf_binary_transformer, BINARY_FEATURES),
        ('cat', rf_categorical_transformer, CATEGORICAL_FEATURES)
    ],
    remainder='drop'
)

# Random Forest classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Random Forest pipeline (Age imputer -> ColumnTransformer -> Classifier)
rf_pipeline = Pipeline(steps=[
    ('age_imputer', rf_age_imputer),
    ('preprocessor', rf_column_transformer),
    ('classifier', rf_classifier)
])


# =============================================================================
# LightGBM Components
# - GroupedMedianAgeImputer for Age (first step)
# - No scaling for numerical
# - Binary to 0/1
# - OrdinalEncoder for categorical + mark as categorical
# =============================================================================

# Age imputer (applied first to full DataFrame)
lgbm_age_imputer = GroupedMedianAgeImputer()

# Numeric transformer: impute with median, no scaling
lgbm_numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Binary transformer: impute with most_frequent, encode to 0/1
lgbm_binary_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

# Categorical transformer: impute with most_frequent, ordinal encode
lgbm_categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# LightGBM column transformer
lgbm_column_transformer = ColumnTransformer(
    transformers=[
        ('num', lgbm_numeric_transformer, TREE_NUMERIC_FEATURES),
        ('bin', lgbm_binary_transformer, BINARY_FEATURES),
        ('cat', lgbm_categorical_transformer, CATEGORICAL_FEATURES)
    ],
    remainder='drop'
)

# LightGBM classifier
# Note: categorical_feature doesn't work with sklearn pipelines (ColumnTransformer outputs numpy arrays)
# The OrdinalEncoder already encodes categoricals as integers which LightGBM handles well
lgbm_classifier = lgb.LGBMClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# LightGBM pipeline (Age imputer -> ColumnTransformer -> Classifier)
lgbm_pipeline = Pipeline(steps=[
    ('age_imputer', lgbm_age_imputer),
    ('preprocessor', lgbm_column_transformer),
    ('classifier', lgbm_classifier)
])


# =============================================================================
# Pipeline Registry
# =============================================================================

PIPELINES = {
    'logistic_regression': lr_pipeline,
    'random_forest': rf_pipeline,
    'lightgbm': lgbm_pipeline
}

PREPROCESSORS = {
    'logistic_regression': lr_column_transformer,
    'random_forest': rf_column_transformer,
    'lightgbm': lgbm_column_transformer
}

CLASSIFIERS = {
    'logistic_regression': lr_classifier,
    'random_forest': rf_classifier,
    'lightgbm': lgbm_classifier
}

IMPUTERS = {
    'age': GroupedMedianAgeImputer
}