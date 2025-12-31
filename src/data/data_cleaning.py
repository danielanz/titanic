"""
Data processing script for Titanic dataset.
Performs cleaning on train and test datasets.
"""
import pandas as pd
from pathlib import Path


def convert_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Convert appropriate columns to categorical types."""
    # Pclass as ordinal categorical
    df["Pclass"] = pd.Categorical(
        df["Pclass"],
        categories=[1, 2, 3],
        ordered=True
    )
    
    # Other columns as nominal categorical
    df["Embarked"] = pd.Categorical(df["Embarked"])
    return df


def convert_binary_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """Convert binary columns to integer types."""
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype(int)
    return df

def process_train_data(raw_path: Path) -> pd.DataFrame:
    """Process training data with all cleaning and imputation steps."""
    print("Loading train data...")
    train_df = pd.read_csv(raw_path / "train.csv")
    
    print("Converting to categorical types...")
    train_df = convert_to_categorical(train_df)

    print("Converting binary columns to integer types...")
    train_df = convert_binary_to_int(train_df)
    
    print(f"Train data processed: {train_df.shape}")
    print(f"Missing values:\n{train_df.isnull().sum()}")
    print(f"Train data columns:\n{train_df.columns.tolist()}")
    print(f"Train data types:\n{train_df.dtypes.to_dict()}")
    
    return train_df


def process_test_data(raw_path: Path, train_df: pd.DataFrame) -> pd.DataFrame:
    """Process test data with all cleaning and imputation steps."""
    print("\nLoading test data...")
    test_df = pd.read_csv(raw_path / "test.csv")
    
    print("Converting to categorical types...")
    test_df = convert_to_categorical(test_df)
    
    print("Converting binary columns to integer types...")
    test_df = convert_binary_to_int(test_df)
    
    print(f"Test data processed: {test_df.shape}")
    print(f"Missing values:\n{test_df.isnull().sum()}")
    print(f"Test data columns:\n{test_df.columns.tolist()}")
    print(f"Test data types:\n{test_df.dtypes.to_dict()}")

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

