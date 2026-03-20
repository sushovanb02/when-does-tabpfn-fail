from data_loader import load_dataset, subsample_data, add_label_noise
from models import get_models, train_and_evaluate
from experiments.small_data import run_small_data_experiment, plot_results, run_small_data_experiment, plot_results as plot_small
from experiments.noise import run_noise_experiment, plot_results as plot_noise

# Load dataset
X_train, X_test, y_train, y_test = load_dataset("breast_cancer")

print("Train shape:", X_train.shape)

# Subsample
X_small, y_small = subsample_data(X_train, y_train, 100)
print("Subsample shape:", X_small.shape)

# Add noise
y_noisy = add_label_noise(y_small, 0.2)
print("Original labels:", y_small[:10])
print("Noisy labels:", y_noisy[:10])

# Get models
models = get_models()

# Train & evaluate
results = train_and_evaluate(models, X_train, y_train, X_test, y_test)

print("\nFinal Results:", results)

# Run experiment
sizes, results = run_small_data_experiment("breast_cancer")

# Plot results
plot_results(sizes, results)

# Small data experiment
sizes, results_small = run_small_data_experiment("breast_cancer")
plot_small(sizes, results_small)

# Noise experiment
noise_levels, results_noise = run_noise_experiment("breast_cancer")
plot_noise(noise_levels, results_noise)