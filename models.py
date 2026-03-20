from tabpfn import TabPFNClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


def get_models():
    """
    Initialize all models.
    """

    models = {
        "TabPFN": TabPFNClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    return models


def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    """
    Train all models and return accuracy results.
    """

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        acc = accuracy_score(y_test, y_pred)

        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")

    return results