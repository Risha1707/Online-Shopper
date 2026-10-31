# Online Shoppers Purchase Intention Prediction

## 1. Project Overview

This project uses machine learning classification algorithms to predict whether an online shopping visitor will make a purchase during a browsing session. It uses the Online Shoppers Purchasing Intention dataset.

The target column is `Revenue`. The model predicts one of two outcomes:
- `No Purchase` — the session did not result in a purchase (`Revenue = False`).
- `Purchase` — the session resulted in a purchase (`Revenue = True`).

The project trains and compares five classification models: K-Nearest Neighbors (KNN), Gaussian Naive Bayes, Decision Tree, Support Vector Machine (SVM), and Random Forest.

## 2. Project Objectives

- Load and inspect the Online Shoppers Purchasing Intention dataset.
- Select relevant numerical and categorical features.
- Remove duplicate records and handle missing values.
- Explore shopping-session data through visualizations.
- Predict purchase intention using five classification algorithms.
- Evaluate the models using accuracy, precision, recall, and F1-score.
- Compare model performance and visualize important features.

## 3. Dataset Description

**Dataset file:** `online_shoppers_intention.csv`

**Target column:** `Revenue`

The code uses the following input features.

### Numerical columns
- `Administrative`
- `Administrative_Duration`
- `Informational`
- `Informational_Duration`
- `ProductRelated`
- `ProductRelated_Duration`
- `BounceRates`
- `ExitRates`
- `PageValues`
- `SpecialDay`
- `OperatingSystems`
- `Browser`
- `Region`
- `TrafficType`

### Categorical columns
- `Month`
- `VisitorType`
- `Weekend`

The target column `Revenue` is converted into the labels `No Purchase` and `Purchase`. It is kept separate from the input features to avoid using the answer as a model input.

## 4. Data Preprocessing

The Python script performs these preprocessing steps:

1. Loads the CSV file using pandas.
2. Selects the required feature columns and the `Revenue` target.
3. Removes duplicate rows.
4. Converts numerical columns to numeric data types.
5. Fills missing numerical values with each column's median.
6. Fills missing categorical values with each column's mode.
7. Converts the `Revenue` values into readable purchase categories.
8. Applies one-hot encoding to categorical input features using `pd.get_dummies()`.

## 5. Exploratory Data Analysis (EDA)

The script displays four visualizations:

1. **Top Visitor Types:** bar chart showing the most frequent visitor types.
2. **Purchase Intention Distribution:** pie chart showing the proportion of purchase and no-purchase sessions.
3. **Page Values Distribution:** histogram showing the distribution of `PageValues`.
4. **Product Related Pages vs Page Values:** scatter plot comparing `ProductRelated` with `PageValues`.

## 6. Machine Learning Models

The project uses exactly five classification models:

1. **K-Nearest Neighbors (KNN):** classifies a session based on nearby training examples.
2. **Gaussian Naive Bayes:** uses probability and the Naive Bayes assumption to classify sessions.
3. **Decision Tree:** predicts the class using a tree of feature-based decisions.
4. **Support Vector Machine (SVM):** finds a decision boundary between the classes.
5. **Random Forest:** combines multiple decision trees to make a prediction.

### Training and testing

- Training data: 70%
- Testing data: 30%
- `random_state = 42`
- `stratify = y` is used to preserve the class distribution in the train and test sets.
- `StandardScaler` is used for KNN and SVM.
- The other three models use the one-hot-encoded, unscaled features.

## 7. Model Evaluation

Each model is evaluated using the following metrics:

- **Accuracy:** the proportion of all predictions that are correct.
- **Precision:** the proportion of predicted class labels that are correct.
- **Recall:** the proportion of actual class labels correctly identified.
- **F1-score:** the harmonic mean of precision and recall.

The script calculates weighted precision, weighted recall, and weighted F1-score. It also prints a classification report for each model and displays a confusion matrix for each model.

## 8. Model Comparison

The evaluation results for all five models are collected in a pandas DataFrame with these columns:

- `Model`
- `Accuracy`
- `Precision`
- `Recall`
- `F1 Score`

The script prints the comparison table sorted by accuracy and displays a bar chart comparing the accuracy of the five models.

## 9. Decision Tree Visualization

The trained Decision Tree is visualized using `plot_tree()`. The displayed tree is limited to a maximum depth of 3 for readability and shows the feature names and the two class names: `No Purchase` and `Purchase`.

## 10. Random Forest Feature Importance

The script obtains feature importance values from the trained Random Forest model. It displays a bar chart of the top 10 most important input features.

## 11. Technologies and Libraries

- Python
- pandas
- matplotlib
- scikit-learn

## 12. Installation

Install the required libraries by running:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain:

```text
pandas
matplotlib
scikit-learn
```

## 13. How to Run the Project

1. Keep `PredictiveProject.py`, `README.md`, `requirements.txt`, and `online_shoppers_intention.csv` in the project folder.
2. Open the Python file in Spyder, VS Code, or another Python IDE.
3. Make sure the CSV path in `pd.read_csv(...)` points to the actual location of `online_shoppers_intention.csv`.
4. Run the Python script.

If the CSV file is in the same folder as the Python script, use:

```python
df = pd.read_csv("online_shoppers_intention.csv")
```

If the file is stored elsewhere, replace the path with its actual location.

## 14. Project Outputs

The script prints:
- Dataset shape and first five rows.
- Purchase and no-purchase class counts.
- Accuracy, precision, recall, and F1-score for each model.
- A classification report for each model.
- A model comparison table sorted by accuracy.

The script displays:
- Top visitor types bar chart.
- Purchase intention pie chart.
- Page values histogram.
- Product-related pages versus page values scatter plot.
- Five confusion matrices, one for each model.
- Accuracy comparison bar chart.
- Decision Tree visualization.
- Top 10 Random Forest feature-importance chart.

**Note:** This version displays the results and charts. It does not save model metrics or feature importance to separate CSV files.

## 15. Repository Structure

```text
Online_Shoppers_Project/
|
|-- PredictiveProject.py
|-- README.md
|-- requirements.txt
|-- online_shoppers_intention.csv
```

## 16. Conclusion

This project applies five classification algorithms to the Online Shoppers Purchasing Intention dataset to predict whether a visitor session will result in a purchase. It includes data cleaning, exploratory visualizations, model training, evaluation, model comparison, Decision Tree visualization, and Random Forest feature-importance analysis. The evaluation metrics help compare the models' performance on the test data.
