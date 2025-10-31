import pandas as pd
from sklearn.model_selection import StratifiedKFold # Although ESC-50 is balanced, this is good practice
import numpy as np

def get_train_val_test_splits(metadata, num_folds=5, current_test_fold_idx=0):
    """
    Generates train, validation, and test splits based on ESC-50's pre-defined folds.
    
    Args:
        metadata (pd.DataFrame): The ESC-50 metadata DataFrame.
        num_folds (int): Total number of folds in the dataset (e.g., 5 for ESC-50).
        current_test_fold_idx (int): The index (0-based) of the fold to be used as the test set
                                     in the current cross-validation iteration.

    Returns:
        tuple: (train_df, val_df, test_df) - DataFrames for each split.
    """
    all_folds = sorted(metadata['fold'].unique()) # Get unique fold numbers: [1, 2, 3, 4, 5]

    if not (0 <= current_test_fold_idx < num_folds):
        raise ValueError(f"current_test_fold_idx must be between 0 and {num_folds-1}")

    test_fold = all_folds[current_test_fold_idx]
    
    remaining_folds = [f for f in all_folds if f != test_fold]
    
    # Arbitrarily pick one of the remaining folds for validation, others for training
    # For a full CV, this would cycle. Here, we demonstrate one such split.
    val_fold = remaining_folds.pop(0) # Take the first remaining fold as validation
    train_folds = remaining_folds # The rest are training folds

    print(f"--- Data Split for Test Fold {test_fold} ---")
    print(f"Train Folds: {train_folds}")
    print(f"Validation Fold: {val_fold}")
    print(f"Test Fold: {test_fold}")

    train_df = metadata[metadata['fold'].isin(train_folds)].reset_index(drop=True)
    val_df = metadata[metadata['fold'] == val_fold].reset_index(drop=True)
    test_df = metadata[metadata['fold'] == test_fold].reset_index(drop=True)

    # Optional: Verify class distribution if the dataset was not perfectly balanced
    # print("\nTrain Class Distribution:\n", train_df['category'].value_counts(normalize=True))
    # print("Validation Class Distribution:\n", val_df['category'].value_counts(normalize=True))
    # print("Test Class Distribution:\n", test_df['category'].value_counts(normalize=True))

    return train_df, val_df, test_df

# Example Usage (in a hypothetical main training script or notebook):
if __name__ == '__main__':
    from data_processing import load_metadata
    metadata = load_metadata()

    # Demonstrate one specific split (e.g., using fold 5 as test)
    train_df, val_df, test_df = get_train_val_test_splits(metadata, current_test_fold_idx=4) # 4 means fold 5

    print(f"\nTrain set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Total samples: {len(train_df) + len(val_df) + len(test_df)} (should be 2000)")

    # You would then loop through current_test_fold_idx from 0 to 4 for full 5-fold CV