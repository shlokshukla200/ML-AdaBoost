# ⚡ AdaBoost from Scratch using Python

🚀 **A simple yet powerful implementation of the AdaBoost algorithm built completely from scratch using NumPy and Scikit-learn for comparison.**  
This project demonstrates how AdaBoost works internally — from creating weak learners (Decision Stumps) to combining them into a strong ensemble classifier.

---

## 🧠 What is AdaBoost?

**AdaBoost (Adaptive Boosting)** is one of the first boosting algorithms invented.  
It combines multiple weak learners (usually simple classifiers like decision stumps 🌳) to create a powerful ensemble model.

Each weak learner focuses on the mistakes made by the previous ones, thereby improving overall accuracy iteratively.

---

## 🧩 Project Structure

├── adaboost_from_scratch.py # Main Python file containing all code
├── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Libraries Used

- 🧮 **NumPy** — for mathematical operations and array manipulation  
- 🧠 **Scikit-learn** — for dataset generation, model comparison, and evaluation  

---

## 🏗️ Classes Overview

### 🪓 `DecisionStump`
A weak learner used by AdaBoost.  
It finds the best threshold and feature to split data to minimize classification error.

### 🧱 `AdaBoost`
The main AdaBoost class that:
1. Trains multiple Decision Stumps iteratively  
2. Assigns weights to each weak learner based on their performance  
3. Combines all weak learners for final predictions  

---

## 💻 Example Usage

```python
if __name__ == "__main__":
    print("Running AdaBoost from scratch...")

    # Generate synthetic data
    X, y = make_classification(n_samples=200, n_features=2,
                               n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)

    # Convert labels {0,1} → {-1,1}
    y = np.where(y == 0, -1, 1)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Train custom AdaBoost
    adaboost = AdaBoost(n_estimators=5)
    adaboost.fit(X_train, y_train)

    # Predictions & Accuracy
    y_pred = adaboost.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Custom AdaBoost Accuracy: {accuracy:.4f}")
