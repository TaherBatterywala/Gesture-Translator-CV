import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


def main():
    data_path = os.path.join(os.path.dirname(__file__), "../data/processed/dataset.csv")
    model_path = os.path.join(os.path.dirname(__file__), "../models/gesture_model.pkl")

    print("Loading data from:", data_path)
    if not os.path.exists(data_path):
        print(
            f"Error: {data_path} not found. Please run collect_data.py first to collect some data."
        )
        return

    df = pd.read_csv(data_path)
    if df.empty or "label" not in df.columns:
        print("Error: Dataset is empty or incorrectly formatted.")
        return

    X = df.drop("label", axis=1)
    y = df["label"]

    print(f"Dataset Details: {len(X)} samples with {X.shape[1]} features each.")
    print(f"Unique classes: {y.unique()}")

    # Given this is BYOP, we set a reproducible random state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and Train Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Training Random Forest Classifier...")
    rf_clf.fit(X_train, y_train)

    # Evaluate Model
    y_pred = rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Feature Importance for the Project Report
    importances = list(zip(X.columns, rf_clf.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 5 Important Features:")
    for name, imp in importances[:5]:
        print(f" - {name}: {imp:.4f}")

    # Save Model
    print(f"\nSaving model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(rf_clf, f)
    print("Training Complete!")


if __name__ == "__main__":
    main()
