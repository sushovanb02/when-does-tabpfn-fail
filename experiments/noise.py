import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_dataset, add_label_noise
from models import get_models, train_and_evaluate


def run_noise_experiment(dataset_name="breast_cancer"):
    """
    Evaluate models under different levels of label noise.
    """

    # Load dataset
    X_train, X_test, y_train, y_test = load_dataset(dataset_name)

    # Noise levels
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]

    all_results = {
        "TabPFN": [],
        "LogisticRegression": [],
        "XGBoost": []
    }

    for noise in noise_levels:
        print(f"\n--- Training with {int(noise*100)}% label noise ---")

        # Add noise to training labels
        y_noisy = add_label_noise(y_train, noise)

        models = get_models()
        results = train_and_evaluate(models, X_train, y_noisy, X_test, y_test)

        for model_name in all_results.keys():
            all_results[model_name].append(results[model_name])

    return noise_levels, all_results


def plot_results(noise_levels, results):
    """
    Plot Accuracy vs Noise Level
    """

    for model_name, acc_list in results.items():
        plt.plot(noise_levels, acc_list, marker='o', label=model_name)

    plt.xlabel("Noise Level")
    plt.ylabel("Accuracy")
    plt.title("Performance vs Label Noise")
    plt.legend()
    plt.grid()

    plt.savefig("results/noise_plot.png")
    plt.show()