"""
Data processing script for Titanic dataset.
Performs cleaning and imputation on train and test datasets.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def create_title_column(df: pd.DataFrame) -> pd.DataFrame:
    """Extract title from Name column."""
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    return df


def create_has_cabin_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary feature indicating if passenger had a cabin."""
    df["has_cabin"] = df["Cabin"].notna().astype(int)
    return df


def drop_unnecessary_columns(df: pd.DataFrame, keep_passenger_id: bool = False) -> pd.DataFrame:
    """Drop columns that won't be used for modeling."""
    cols_to_drop = ["Name", "Ticket", "Cabin"]
    if not keep_passenger_id and "PassengerId" in df.columns:
        cols_to_drop.append("PassengerId")
    df = df.drop(columns=cols_to_drop)
    return df


def convert_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Convert appropriate columns to categorical types."""
    # Pclass as ordinal categorical
    df["Pclass"] = pd.Categorical(
        df["Pclass"],
        categories=[1, 2, 3],
        ordered=True
    )
    
    # Other columns as nominal categorical
    df["Sex"] = pd.Categorical(df["Sex"])
    df["Embarked"] = pd.Categorical(df["Embarked"])
    df["Title"] = pd.Categorical(df["Title"])
    
    return df


def fill_missing_age_train(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Age values using group-based median imputation."""
    df["Age"] = df["Age"].fillna(
        df.groupby(["Sex", "Pclass", "Title"], observed=False)["Age"].transform("median")
    )
    return df


def fill_missing_embarked_train(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Embarked values with mode."""
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    return df


def fill_missing_age_test(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill missing Age values in test data using train data group medians.
    Falls back to global train median if group not found.
    """
    age_group_cols = ["Sex", "Pclass", "Title"]
    age_medians = train_df.groupby(age_group_cols, observed=False)["Age"].median()
    global_age_median = train_df["Age"].median()
    
    def fill_age(row):
        if pd.isnull(row["Age"]):
            group_key = (row["Sex"], row["Pclass"], row["Title"])
            group_median = age_medians.get(group_key, np.nan)
            if pd.isnull(group_median):
                return global_age_median
            else:
                return group_median
        else:
            return row["Age"]
    
    test_df["Age"] = test_df.apply(fill_age, axis=1)
    return test_df


def fill_missing_fare_test(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame
) -> pd.DataFrame:
    """Fill missing Fare values in test data using train data median."""
    test_df["Fare"] = test_df["Fare"].fillna(train_df["Fare"].median())
    return test_df


def process_train_data(raw_path: Path) -> pd.DataFrame:
    """Process training data with all cleaning and imputation steps."""
    print("Loading train data...")
    train_df = pd.read_csv(raw_path / "train.csv")
    
    print("Creating Title column...")
    train_df = create_title_column(train_df)
    
    print("Creating has_cabin feature...")
    train_df = create_has_cabin_feature(train_df)
    
    print("Dropping unnecessary columns...")
    train_df = drop_unnecessary_columns(train_df, keep_passenger_id=False)
    
    print("Converting to categorical types...")
    train_df = convert_to_categorical(train_df)
    
    print("Filling missing Age values...")
    train_df = fill_missing_age_train(train_df)
    
    print("Filling missing Embarked values...")
    train_df = fill_missing_embarked_train(train_df)
    
    print(f"Train data processed: {train_df.shape}")
    print(f"Missing values:\n{train_df.isnull().sum()}")
    
    return train_df


def process_test_data(raw_path: Path, train_df: pd.DataFrame) -> pd.DataFrame:
    """Process test data with all cleaning and imputation steps."""
    print("\nLoading test data...")
    test_df = pd.read_csv(raw_path / "test.csv")
    
    print("Creating Title column...")
    test_df = create_title_column(test_df)
    
    print("Creating has_cabin feature...")
    test_df = create_has_cabin_feature(test_df)
    
    print("Dropping unnecessary columns...")
    test_df = drop_unnecessary_columns(test_df, keep_passenger_id=False)
    
    print("Converting to categorical types...")
    test_df = convert_to_categorical(test_df)
    
    print("Filling missing Age values (using train data statistics)...")
    test_df = fill_missing_age_test(test_df, train_df)
    
    print("Filling missing Fare values (using train data median)...")
    test_df = fill_missing_fare_test(test_df, train_df)
    
    print(f"Test data processed: {test_df.shape}")
    print(f"Missing values:\n{test_df.isnull().sum()}")
    
    return test_df


def main():
    """Main function to process and save datasets."""
    # Define paths
    raw_dir = Path("input/raw")
    processed_dir = Path("input/processed")
    
    # Process train data
    train_processed = process_train_data(raw_dir)
    
    # Process test data (using train data for imputation reference)
    test_processed = process_test_data(raw_dir, train_processed)
    
    # Save processed datasets
    train_output = processed_dir / "train_processed.parquet"
    test_output = processed_dir / "test_processed.parquet"
    
    print(f"\nSaving processed train data to {train_output}...")
    train_processed.to_parquet(train_output, index=False, engine='fastparquet')
    
    print(f"Saving processed test data to {test_output}...")
    test_processed.to_parquet(test_output, index=False, engine='fastparquet')
    
    print("\n✓ Data processing complete!")


if __name__ == "__main__":
    main()

