# Logistic Regression Implementation in Python

A hands-on exploration of binary classification using **Logistic Regression** with scikit-learn. This repo contains two Jupyter notebooks demonstrating the full ML pipeline on different datasets.

---

## 📓 Notebooks

### 1. [`load_breast_cancer.ipynb`](./load_breast_cancer.ipynb)
Applies logistic regression to the classic **Wisconsin Breast Cancer** dataset from scikit-learn.

**Pipeline:**
- Loads the dataset and converts it to a pandas DataFrame
- Performs an 80/20 train-test split (`random_state=42`)
- Scales features using `StandardScaler`
- Trains a `LogisticRegression` model (`max_iter=1000`)
- Generates predictions on the test set

**Dataset:** 569 samples × 30 features (cell nucleus measurements), binary target (malignant / benign)

---

### 2. [`make_classification.ipynb`](./make_classification.ipynb)
Applies logistic regression to a **synthetically generated** binary classification dataset.

**Pipeline:**
- Generates a dataset using `sklearn.datasets.make_classification`
- 1,000 samples × 10 features, 2 classes (`random_state=42`)
- Performs an 80/20 train-test split (`random_state=42`)
- Trains a `LogisticRegression` model
- Evaluates performance with accuracy score and classification report
- Visualises results with a confusion matrix heatmap

---

## 🛠️ Requirements

| Package | Purpose |
|---|---|
| `scikit-learn` | Datasets, model, preprocessing, train/test split |
| `pandas` | DataFrame handling |
| `numpy` | Numerical operations |
| `matplotlib` | Visualisation |
| `seaborn` | Visualisation |

Install all dependencies with:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

---

## 🚀 Getting Started

1. **Clone the repo**
   ```bash
   git clone <your-repo-url>
   cd logistic_regression_implementation
   ```

2. **Install dependencies**
   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn
   ```

3. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```

4. Open either notebook and run all cells.

---

## 📖 What is Logistic Regression?

Logistic Regression is a supervised learning algorithm for **binary classification**. Despite its name, it is a classification model — not a regression model. It estimates the probability that an input belongs to a particular class using the **sigmoid function**:

$$P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \cdots + \beta_n X_n)}}$$

Key characteristics:
- Outputs a probability between 0 and 1
- Decision boundary is linear
- Works best when features are scaled (hence the use of `StandardScaler`)
- Fast to train and highly interpretable
