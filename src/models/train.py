"""
Training script for Titanic classification models.
Loads train features, tunes hyperparameters with GridSearchCV, and saves trained models.
"""
import warnings
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

from pipelines import (
    lr_pipeline,
    rf_pipeline,
    lgbm_pipeline
)


# =============================================================================
# Hyperparameter Grids (start small, tune iteratively)
# =============================================================================

LR_PARAM_GRID = {
    'classifier__C': [5, 7, 10, 20, 30, 40, 50],
    'classifier__l1_ratio': [0, 0.25, 0.5, 0.75, 1],
}

RF_PARAM_GRID = {
    'classifier__n_estimators': [300, 400, 500],
    'classifier__max_depth': [None, 5, 10, 15, 20],
    'classifier__min_samples_leaf': [1, 2, 3, 4, 5],
    'classifier__min_samples_split': [5, 7, 10]
}

LGBM_PARAM_GRID = {
    'classifier__n_estimators': [500, 700, 900],
    'classifier__num_leaves': [31, 63],
    'classifier__max_depth': [3, 5, -1],
    'classifier__reg_alpha': [0, 1],
    'classifier__reg_lambda': [0, 1],
    'classifier__learning_rate': [0.01, 0.02],
}


def load_train_features(features_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load training features and separate into X and y.
    
    Args:
        features_path: Path to the train features parquet file
        
    Returns:
        Tuple of (X features DataFrame, y target Series)
    """
    train_df = pd.read_parquet(features_path)
    y = train_df['survived']
    X = train_df.drop(columns=['survived'])
    return X, y


def train_with_gridsearch(
    pipeline,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    output_dir: Path,
    cv: int = 5
) -> GridSearchCV:
    """
    Train a pipeline using GridSearchCV and save the best model.
    
    Args:
        pipeline: sklearn Pipeline to tune
        param_grid: Dictionary of hyperparameters to search
        X: Features DataFrame
        y: Target Series
        model_name: Name for the saved model file
        output_dir: Directory to save the model
        cv: Number of cross-validation folds
        
    Returns:
        Fitted GridSearchCV object
    """
    print(f"\nTraining {model_name} with GridSearchCV...")
    print(f"  Parameter grid: {param_grid}")
    
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X, y)
    
    print(f"\n  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV accuracy: {grid_search.best_score_:.4f}")
    
    # Save the best model
    model_path = output_dir / f"{model_name}.joblib"
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"  ✓ Saved best model to {model_path}")
    
    return grid_search


def main():
    """Main function to tune and save all models."""
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    features_path = project_root / "input" / "features" / "train_features.parquet"
    models_dir = project_root / "output" / "models"
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    print("Loading training features...")
    X, y = load_train_features(features_path)
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    
    # Train each model with GridSearchCV
    print("\n" + "=" * 60)
    print("Training models with GridSearchCV (cv=5)")
    print("=" * 60)
    
    results = {}
    
    results['Logistic Regression'] = train_with_gridsearch(
        lr_pipeline, LR_PARAM_GRID, X, y, "logistic_regression", models_dir
    )
    
    results['Random Forest'] = train_with_gridsearch(
        rf_pipeline, RF_PARAM_GRID, X, y, "random_forest", models_dir
    )
    
    # results['LightGBM'] = train_with_gridsearch(
    #     lgbm_pipeline, LGBM_PARAM_GRID, X, y, "lightgbm", models_dir
    # )
    
    # Print final summary
    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)
    print(f"\n{'Model':<25} {'Best CV Accuracy':<18} {'Best Parameters'}")
    print("-" * 80)
    for name, grid_search in results.items():
        params_str = ', '.join(f"{k.split('__')[1]}={v}" for k, v in grid_search.best_params_.items())
        print(f"{name:<25} {grid_search.best_score_:.4f}            {params_str}")
    
    # Find best model
    best_model = max(results, key=lambda x: results[x].best_score_)
    print("\n" + "=" * 60)
    print(f"✓ Best Model: {best_model} (CV Accuracy: {results[best_model].best_score_:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
