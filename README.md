# HackTheFest AI Bias Hackathon

## Project Overview

This project was developed for the AI Bias Hackathon to investigate, test, detect and report algorithimic bias and fairness in a loan approval model. Using SHAP, Fairness Metrics and Visualization Techniques, We evaluated potential bias and discrimination in the dataset with a focus on sensitive attributes like gender and race.

## Problem Statement

The financial sector has long faced scrutiny over systemic inequities embedded in its decision-making processes, particularly in loan approvals. This project explores the simulation of the real world in detection and explantation of unusual patterns in AI decision making for mortage loan approvals using the provided dataset.

## Project structure

```
├── LICENSE
├── README.md                      # Project Documentation
├── bias_visualization.png         # Bias visualization
├── confusion_matrix.png           # Confusion matrix visualization
├── datasets                       # Datasets
│   ├── loan_access_dataset.csv
│   ├── loan_access_dataset.xlsx
│   └── test.csv
├── feature_importance.png         # Feature importance visualization
├── loan_model.ipynb               # Jupiter notebook
├── loan_model.py                  # Model trainining and bias analysis script
├── requirements.txt               # Dependency list
├── shap_summary.png               # SHAP value summary plot
└── submission.csv                 # Final Output files

```

## Dataset

- **loan_access_dataset.csv:** Labeled training data
- **test.csv:** Unlabeled data for model predictions.

Sensitve attributes audited:

- Gender
- Race
- Age
- Employment Status
- Education
- Zip Code Group

## Methodology and Approach

- **Data Loading and Prepocessing**

  - Cleaned and prepare the dataset for modelling
  - Encoded categorical variables

- **Model Training**

  - **XGBoost Classifier:** choosen for its balance in performance and interpretability
  - Key parameters: `max_depth`, `learning_rate`, `n_estimators`

- **Bias Detection**

  - Conducted SHAP value analytics for feature importance and potential discriminatory attributes.
  - Group wise approval rate analysis

- **Visual Analysis**
  - Generated visual evidence of bias:
    - Feature importance
    - Confusion matrix analysis
    - SHAP summary plots
    - Bias visualization

## Key Findings

- **Gender Bias:**
  Male applicants had ~15% higher approval rates

- **Racial Bias:**
  Notable disparity in approval rate between different racial groups

- **Geographic Bias:** Zip code disparities indicate location-based bias

- **Education disparity:** Lower approval rates for less educated applicants

## Visuals

- ### APPROVAL RATE COMPARISONS ACROSS DEMOGRAPHICS GROUPS

  ![Bias Visualization](bias_visualization.png)

  ### SHAP SUMMARY PLOTS

  ![Bias Visualization](shap_summary.png)

- ### CONFUSION MATRIX ANALYSIS

  ![Confusion Matrix](confusion_matrix.png)

- ### FEATURE IMPORTANCES
  ![Feature Importance](feature_importance.png)

## Tools and Libraries

Before you begin, ensure you have met the following requirements:

- Python 3.9+
- Pandas
- Scikit-learn
- SHAP
- Matplotib
- Seaborn
- XGBoost

## Running the project

1. Clone the Repository

```
git clone https://github.com/imisi99/ai-bias-submission.git
cd ai-bias-submission
```

2. Create a virtual environment

```
python -m venv venv
source venv/bin/activate
```

3. Install required dependencies

```
pip install -r requirements.txt
```

4. Run Script

```
python loan_model.py
```

---

_This project was built as part of the AI Bias Bounty Hackathon._
