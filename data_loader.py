import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(name="breast_cancer", test_size=0.2, random_state=42):
    """
    Load and preprocess dataset.
    """

    if name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)

    elif name == "iris":
        X, y = load_iris(return_X_y=True)

    else:
        raise ValueError(f"Dataset {name} not supported")

    return preprocess_data(X, y, test_size, random_state)


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Standard preprocessing:
    - Train/test split
    - Feature scaling
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def subsample_data(X, y, n_samples, random_state=42):
    """
    Subsample dataset for small-data experiments.
    """
    np.random.seed(random_state)
    indices = np.random.choice(len(X), size=n_samples, replace=False)
    return X[indices], y[indices]


def add_label_noise(y, noise_ratio, random_state=42):
    """
    Add label noise by flipping labels.
    """
    np.random.seed(random_state)
    y_noisy = y.copy()

    n_noisy = int(len(y) * noise_ratio)
    noisy_indices = np.random.choice(len(y), size=n_noisy, replace=False)

    unique_classes = np.unique(y)

    for idx in noisy_indices:
        y_noisy[idx] = np.random.choice(unique_classes)

    return y_noisy