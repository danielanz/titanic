"""
Feature engineering script for Titanic dataset.
Creates additional features from processed data.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def create_is_female_or_child(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary feature indicating if passenger is female or child (age < 16).
    
    Args:
        df: DataFrame with 'Sex' and 'Age' columns
        
    Returns:
        DataFrame with 'is_female_or_child' column added
    """
    df['is_female_or_child'] = ((df['Sex'] == 'female') | (df['Age'] < 16)).astype(int)
    return df


def create_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create family_size feature from SibSp and Parch.
    family_size = SibSp + Parch + 1 (includes the passenger)
    
    Args:
        df: DataFrame with 'SibSp' and 'Parch' columns
        
    Returns:
        DataFrame with 'family_size' column added
    """
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    return df


def create_is_alone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary feature indicating if passenger is alone (family_size == 1).
    
    Args:
        df: DataFrame with 'family_size' column
        
    Returns:
        DataFrame with 'is_alone' column added
    """
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    return df


def create_sex_pclass_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction feature between Sex and Pclass.
    Useful for linear models to capture the interaction effect.
    
    Args:
        df: DataFrame with 'Sex' and 'Pclass' columns
        
    Returns:
        DataFrame with 'sex_pclass' column added as categorical
    """
    df['sex_pclass'] = df['Sex'].astype(str) + '_' + df['Pclass'].astype(str)
    df['sex_pclass'] = pd.Categorical(df['sex_pclass'])
    return df


def create_fare_per_person(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create fare_per_person feature by dividing Fare by family_size.
    
    Args:
        df: DataFrame with 'Fare' and 'family_size' columns
        
    Returns:
        DataFrame with 'fare_per_person' column added
    """
    df['fare_per_person'] = df['Fare'] / df['family_size']
    return df


def drop_redundant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop features that are redundant after creating new features.
    Drops: SibSp, Parch (replaced by family_size/is_alone)
           Fare (replaced by fare_per_person)
    
    Args:
        df: DataFrame with redundant features
        
    Returns:
        DataFrame with redundant features removed
    """
    cols_to_drop = ['SibSp', 'Parch', 'Fare']
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)
    return df


def apply_log_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log transformation to fare_per_person to reduce skewness.
    Uses log1p (log(1 + x)) to handle zero values.
    
    Args:
        df: DataFrame with 'fare_per_person' column
        
    Returns:
        DataFrame with log-transformed 'fare_per_person'
    """
    df['fare_per_person'] = np.log1p(df['fare_per_person'])
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in the correct order.
    
    Args:
        df: Processed DataFrame from make_dataset.py
        
    Returns:
        DataFrame with all engineered features
    """
    # Create new features
    df = create_is_female_or_child(df)
    df = create_family_size(df)
    df = create_is_alone(df)
    df = create_sex_pclass_interaction(df)
    df = create_fare_per_person(df)
    
    # Drop redundant features
    df = drop_redundant_features(df)
    
    # Apply transformations
    df = apply_log_transformation(df)
    
    return df


def main():
    """
    Main function to load processed data, build features, and save.
    """
    # Define paths
    processed_dir = Path("input/processed")
    features_dir = Path("input/features")
    
    # Create features directory if it doesn't exist
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    print("Loading processed data...")
    train_df = pd.read_parquet(
        processed_dir / "train_processed.parquet",
        engine='fastparquet'
    )
    test_df = pd.read_parquet(
        processed_dir / "test_processed.parquet",
        engine='fastparquet'
    )
    
    print(f"Train shape before feature engineering: {train_df.shape}")
    print(f"Test shape before feature engineering: {test_df.shape}")
    
    # Build features
    print("\nBuilding features for train data...")
    train_features = build_all_features(train_df.copy())
    
    print("Building features for test data...")
    test_features = build_all_features(test_df.copy())
    
    print(f"\nTrain shape after feature engineering: {train_features.shape}")
    print(f"Test shape after feature engineering: {test_features.shape}")
    
    print("\nTrain columns:", train_features.columns.tolist())
    print("Test columns:", test_features.columns.tolist())
    
    # Save feature-engineered data
    train_output = features_dir / "train_features.parquet"
    test_output = features_dir / "test_features.parquet"
    
    print(f"\nSaving feature-engineered train data to {train_output}...")
    train_features.to_parquet(train_output, index=False, engine='fastparquet')
    
    print(f"Saving feature-engineered test data to {test_output}...")
    test_features.to_parquet(test_output, index=False, engine='fastparquet')
    
    print("\n✓ Feature engineering complete!")
    
    # Display info
    print("\n=== Train Features Info ===")
    train_features.info()
    print("\n=== Test Features Info ===")
    test_features.info()


if __name__ == "__main__":
    main()

