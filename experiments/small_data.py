import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_dataset, subsample_data
from models import get_models, train_and_evaluate


def run_small_data_experiment(dataset_name="breast_cancer"):
    """
    Run experiment across different dataset sizes.
    """

    # Load full dataset
    X_train, X_test, y_train, y_test = load_dataset(dataset_name)

    # Different training sizes
    sample_sizes = [50, 100, 200, 500, 1000]

    all_results = {
        "TabPFN": [],
        "LogisticRegression": [],
        "XGBoost": []
    }

    for size in sample_sizes:
        print(f"\n--- Training with {size} samples ---")

        # Ensure size is not larger than dataset
        size = min(size, len(X_train))

        X_sub, y_sub = subsample_data(X_train, y_train, size)

        models = get_models()
        results = train_and_evaluate(models, X_sub, y_sub, X_test, y_test)

        for model_name in all_results.keys():
            all_results[model_name].append(results[model_name])

    return sample_sizes, all_results


def plot_results(sample_sizes, results):
    """
    Plot Accuracy vs Dataset Size
    """

    for model_name, acc_list in results.items():
        plt.plot(sample_sizes, acc_list, marker='o', label=model_name)

    plt.xlabel("Training Dataset Size")
    plt.ylabel("Accuracy")
    plt.title("Performance vs Dataset Size")
    plt.legend()
    plt.grid()

    plt.savefig("results/small_data_plot.png")
    plt.show()