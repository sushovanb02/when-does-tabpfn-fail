from tabpfn import TabPFNClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize model
clf = TabPFNClassifier()

# Train
clf.fit(X_train, y_train)

# Evaluate
acc = clf.score(X_test, y_test)
print(f"Accuracy: {acc:.4f}")