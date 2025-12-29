"""
Evaluation script for Titanic classification models.
Uses nested cross-validation with GridSearchCV to evaluate each pipeline.
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.exceptions import ConvergenceWarning

from pipelines import (
    lr_pipeline,
    rf_pipeline,
    lgbm_pipeline
)

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# =============================================================================
# Hyperparameter Grids (same as train.py)
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
    y = train_df['Survived']
    X = train_df.drop(columns=['Survived'])
    return X, y


def evaluate_with_nested_cv(
    pipeline,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    outer_cv: int = 5,
    inner_cv: int = 5
) -> dict:
    """
    Evaluate a pipeline using nested cross-validation with GridSearchCV.
    
    Nested CV structure:
    - Outer loop: Evaluates model generalization performance
    - Inner loop: Tunes hyperparameters using GridSearchCV
    
    Args:
        pipeline: sklearn Pipeline to evaluate
        param_grid: Dictionary of hyperparameters to search
        X: Features DataFrame
        y: Target Series
        outer_cv: Number of outer CV folds (default: 5)
        inner_cv: Number of inner CV folds (default: 5)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Create GridSearchCV object for the inner loop
    inner_cv_strategy = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=inner_cv_strategy,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    # Outer CV loop
    outer_cv_strategy = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=43)
    
    # Perform nested cross-validation
    # Set n_jobs=1 to avoid nested parallelization (GridSearchCV already parallelizes internally)
    scores = cross_val_score(
        grid_search,
        X,
        y,
        cv=outer_cv_strategy,
        scoring='accuracy',
        n_jobs=1,  # Best practice: parallelize at one level only
        error_score='raise'
    )
    
    return {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std()
    }


def main():
    """Main function to evaluate all pipelines using nested CV."""
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    features_path = project_root / "input" / "features" / "train_features.parquet"
    
    # Load training data
    print("Loading training features...")
    X, y = load_train_features(features_path)
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    
    # Define pipelines and param grids to evaluate
    models = {
        'Logistic Regression': (lr_pipeline, LR_PARAM_GRID),
        'Random Forest': (rf_pipeline, RF_PARAM_GRID),
        'LightGBM': (lgbm_pipeline, LGBM_PARAM_GRID)
    }
    
    # Evaluate each pipeline with nested CV
    print("\n" + "=" * 70)
    print("Evaluating models with Nested Cross-Validation")
    print("Outer CV: 5 folds | Inner CV: 5 folds with GridSearchCV")
    print("=" * 70 + "\n")
    
    results = {}
    for name, (pipeline, param_grid) in models.items():
        print(f"Evaluating {name}...")
        print(f"  Hyperparameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")
        result = evaluate_with_nested_cv(pipeline, param_grid, X, y, outer_cv=5, inner_cv=5)
        results[name] = result
        print(f"  Outer fold scores: {result['scores'].round(4)}")
        print(f"  Mean accuracy: {result['mean']:.4f} (+/- {result['std']:.4f})\n")
    
    # Print final summary
    print("=" * 70)
    print("Final Nested CV Accuracy Summary")
    print("=" * 70)
    print(f"{'Model':<25} {'Mean Accuracy':<15} {'Std Dev':<10}")
    print("-" * 60)
    for name, result in results.items():
        print(f"{name:<25} {result['mean']:.4f}          {result['std']:.4f}")
    
    # Find best model
    best_model = max(results, key=lambda x: results[x]['mean'])
    print("\n" + "=" * 70)
    print(f"✓ Best Model: {best_model} (Nested CV Accuracy: {results[best_model]['mean']:.4f})")
    print("=" * 70)
    print("\nNote: Nested CV provides an unbiased estimate of model generalization.")
    print("These scores may differ from train.py which uses simple CV for tuning.")


if __name__ == "__main__":
    main()
