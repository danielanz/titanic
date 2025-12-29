"""
Prediction script for Titanic classification models.
Loads trained models, makes predictions on test set, and saves submission files.
"""
import joblib
import pandas as pd
from pathlib import Path

def load_test_data(
    features_path: Path,
    raw_path: Path
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load test features and PassengerId from original test.csv.
    
    Args:
        features_path: Path to the test features parquet file
        raw_path: Path to the raw test.csv file
        
    Returns:
        Tuple of (X_test DataFrame, PassengerId Series)
    """
    # Load test features
    X_test = pd.read_parquet(features_path)
    
    # Load original test.csv to get PassengerId
    test_raw = pd.read_csv(raw_path)
    passenger_ids = test_raw['PassengerId']
    
    return X_test, passenger_ids


def make_predictions(
    model_path: Path,
    X_test: pd.DataFrame
) -> pd.Series:
    """
    Load a trained model and make predictions.
    
    Args:
        model_path: Path to the saved model (.joblib file)
        X_test: Test features DataFrame
        
    Returns:
        Series of predictions
    """
    model = joblib.load(model_path)
    predictions = model.predict(X_test)
    return pd.Series(predictions, name='Survived')


def save_submission(
    passenger_ids: pd.Series,
    predictions: pd.Series,
    output_path: Path
) -> None:
    """
    Save predictions as a submission CSV file.
    
    Args:
        passenger_ids: PassengerId Series
        predictions: Survived predictions Series
        output_path: Path to save the submission file
    """
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions.astype(int)
    })
    submission.to_csv(output_path, index=False)
    print(f"  ✓ Saved to {output_path}")


def main():
    """Main function to generate predictions for all models."""
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    features_path = project_root / "input" / "features" / "test_features.parquet"
    raw_path = project_root / "input" / "raw" / "test.csv"
    models_dir = project_root / "output" / "models"
    predictions_dir = project_root / "output" / "predictions"
    
    # Create predictions directory if it doesn't exist
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    print("Loading test data...")
    X_test, passenger_ids = load_test_data(features_path, raw_path)
    print(f"  Test features shape: {X_test.shape}")
    print(f"  Number of passengers: {len(passenger_ids)}")
    
    # Define models and their output files
    models = {
        'Logistic Regression': ('logistic_regression.joblib', 'lr_submission.csv'),
        'Random Forest': ('random_forest.joblib', 'rf_submission.csv'),
        'LightGBM': ('lightgbm.joblib', 'lgbm_submission.csv')
    }
    
    # Generate predictions for each model
    print("\n" + "=" * 60)
    print("Generating predictions...")
    print("=" * 60 + "\n")
    
    for name, (model_file, submission_file) in models.items():
        model_path = models_dir / model_file
        output_path = predictions_dir / submission_file
        
        if not model_path.exists():
            print(f"  ⚠ {name}: Model not found at {model_path}")
            continue
        
        print(f"  {name}:")
        predictions = make_predictions(model_path, X_test)
        save_submission(passenger_ids, predictions, output_path)
        
        # Print prediction distribution
        survived_count = predictions.sum()
        total_count = len(predictions)
        print(f"    Predicted survival rate: {survived_count}/{total_count} ({survived_count/total_count*100:.1f}%)\n")
    
    print("=" * 60)
    print("✓ All predictions generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()

