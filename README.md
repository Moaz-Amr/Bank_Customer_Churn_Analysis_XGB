# Customer Churn Prediction using XGBoost

## Short Description

This project aims to predict whether a bank customer will churn (leave the bank) based on their attributes, using the `Churn_Modelling.csv` dataset. It utilizes the XGBoost classification algorithm and includes data preprocessing steps like encoding categorical features.

## Project Overview

The notebook `day6/Churn_Modelling.ipynb` performs the following:

1.  **Load Data:** Loads the customer churn dataset.
2.  **Feature Selection:** Selects relevant features (columns 3 to 12) for prediction.
3.  **Preprocessing:**
    *   Applies Label Encoding to 'Geography' and 'Gender' columns initially.
    *   Sets up a `ColumnTransformer` with `OneHotEncoder` for the 'Geography' column (index 1 within the selected features `x`) to handle categorical data appropriately within a pipeline.
4.  **Train-Test Split:** Splits the data into training and testing sets.
5.  **Model Training:**
    *   Initially trains an XGBoost classifier (`XGBClassifier`) without the full pipeline.
    *   Trains a final XGBoost model within a `Pipeline` that includes the OneHotEncoding preprocessor step.
6.  **Evaluation:**
    *   Evaluates the models using accuracy score and confusion matrix on the test set.
    *   Performs 10-fold cross-validation on the initial model trained on label-encoded data to assess robustness (though evaluation on the pipelined model is the final one shown).

## Dataset

*   **Churn_Modelling.csv:** Contains bank customer data including demographics, account details, and whether the customer has churned (Exited = 1).

## Model Used

*   **XGBoost (Extreme Gradient Boosting):** A powerful and efficient gradient boosting algorithm widely used for classification and regression tasks.

## Requirements

*   Python 3
*   NumPy
*   Pandas
*   Scikit-learn
*   XGBoost
*   Matplotlib (implicitly used via sklearn metrics display, though no plots are explicitly generated in the final code)

## How to Run

1.  Ensure you have the required libraries installed (`pip install numpy pandas scikit-learn xgboost matplotlib`).
2.  Make sure the `Churn_Modelling.csv` file is in the correct path relative to the notebook.
3.  Run the Jupyter Notebook `day6/Churn_Modelling.ipynb`.
