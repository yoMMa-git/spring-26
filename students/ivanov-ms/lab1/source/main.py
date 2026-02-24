import argparse
import numpy as np
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree

from data import run_data_pipeline, load_data_from_csv, train_val_test_split
from models import DecisionTree
from utils import train_eval_model, compare_with_sklearn


def main():
    parser = argparse.ArgumentParser(description='Decision Tree Classification Pipeline')
    parser.add_argument(
        '--mode', type=str, default='full', choices=['full', 'data', 'train'],
        help='Mode to run: "full" (data and train), "data" (only data pipeline), or "train" (only training pipeline)'
    )
    parser.add_argument('--with-plotting', action='store_true', help='Enable plotting')
    parser.add_argument('--missing-rate', type=float, default=0.05,
                        help='Proportion of missing values to introduce per feature (default: 0.05)')
    parser.add_argument('--max-depth', type=int, default=None,
                        help='Maximum depth of the decision tree')
    parser.add_argument('--min-samples-split', type=int, default=2,
                        help='Minimum samples required to split a node')
    parser.add_argument('--prune', action='store_true',
                        help='Enable reduced-error pruning using validation set')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save the processed data')
    parser.add_argument('--data-path', type=str, default='processed_data.csv',
                        help='Path where processed data is saved (for "train" mode)')
    parser.add_argument('--train-size', type=float, default=0.6,
                        help='Proportion of data for training')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='Proportion of data for validation')

    args = parser.parse_args()

    if args.mode in ['full', 'data']:
        prepared_data = run_data_pipeline(
            missing_rate=args.missing_rate,
            random_seed=args.random_seed,
            return_split=args.mode == 'full',
            train_size=args.train_size,
            val_size=args.val_size,
            save_path=args.save_path
        )

        if args.mode == 'data':
            exit()

        X_train, X_val, X_test, y_train, y_val, y_test = prepared_data
    else:
        print("Loading preprocessed data...")
        df = load_data_from_csv(args.data_path)
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            df, train_size=args.train_size, val_size=args.val_size, random_seed=args.random_seed
        )

    # Convert labels to proper format if needed (they should be -1, 1 already)
    print("\nData shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    if X_val is not None:
        print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

    models = {
        "Custom DT": DecisionTree(
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_seed=args.random_seed
        ),
        "Sklearn DT": SklearnDecisionTree(
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            criterion='gini',
            random_state=args.random_seed
        )
    }

    # Train Custom Decision Tree
    print("\nTraining Custom Decision Tree...")
    if X_val is not None and args.prune:
        print("  (with reduced-error pruning)")

    cm_custom = train_eval_model(
        models["Custom DT"], X_train, y_train, X_test, y_test,
        X_val=X_val, y_val=y_val, prune=X_val is not None and args.prune
    )

    tree_depth = models["Custom DT"].get_depth()
    n_nodes = models["Custom DT"].get_n_nodes()
    n_leaves = models["Custom DT"].get_n_leaves()
    print(f"Tree depth: {tree_depth}, nodes: {n_nodes}, leaves: {n_leaves}")

    # Train Sklearn Decision Tree
    print("\nTraining Sklearn Decision Tree...")
    cm_sklearn = train_eval_model(models["Sklearn DT"], X_train, y_train, X_test, y_test)

    tree_depth = models["Sklearn DT"].get_depth()
    n_nodes = models["Sklearn DT"].tree_.node_count
    n_leaves = models["Sklearn DT"].get_n_leaves()
    print(f"Tree depth: {tree_depth}, nodes: {n_nodes}, leaves: {n_leaves}")

    # Compare models
    models_scores = compare_with_sklearn(models, X_test, y_test)

    if args.with_plotting:
        from utils.plotting import plot_confusion_matrix, plot_roc_curve, plot_feature_importances

        plot_confusion_matrix(cm_custom, title="Confusion Matrix (Custom DT)", img_name="cm_custom_dt.png")
        plot_confusion_matrix(cm_sklearn, title="Confusion Matrix (Sklearn DT)", img_name="cm_sklearn_dt.png")

        plot_roc_curve(y_test, models_scores, img_name="roc_curve_comparison_dt.png")

        if models["Custom DT"].feature_importances_ is not None:
            plot_feature_importances(models["Custom DT"].feature_importances_, models["Custom DT"].feature_names_)

    # Print feature importances if available
    if models["Custom DT"].feature_importances_ is not None:
        importances = models["Custom DT"].feature_importances_
        feat_names = models["Custom DT"].feature_names_
        indices = np.argsort(importances)[::-1][:10]
        print("\nTop %d Feature Importances (Custom DT):" % len(indices))
        for idx in indices:
            if feat_names is not None:
                print(f"  Feature {feat_names[idx]} ({idx}): {importances[idx]:.4f}")
            else:
                print(f"  Feature {idx}: {importances[idx]:.4f}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
