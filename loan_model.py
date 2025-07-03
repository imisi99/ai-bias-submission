import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def preprocessing(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes the dataset located at the provided file path by performing missing value handling,
    one-hot encoding, binary encoding, and other transformations. This function returns the
    original dataset and the processed dataset.

    :param filepath: The file path to the dataset in CSV format.
    :type filepath: str
    :return: A tuple containing the original dataset as a pandas DataFrame and the processed
        dataset as another pandas DataFrame.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    
    df = pd.read_csv(filepath)
    print(f"Original dataset shape: {df.shape}")
    
    # Storing original columns for bias analysis
    original_df = df.copy()
    
    # Dropping the ID column
    df = df.drop(columns=['ID', 'Age'])
    
    # Handling the missing values if exists
    num_col = ["Income", "Credit_Score", "Loan_Amount"]
    cat_col = ["Gender", "Race", "Age_Group", "Employment_Type", "Education_Level", "Citizenship_Status", "Zip_Code_Group", "Language_Proficiency"]
    bin_col = ["Disability_Status", "Criminal_Record"]

    # Fill the numerical missing value with the mean
    for col in num_col:
        df[col].fillna(df[col].mean(), inplace=True)

    # Fill the categorical missing value with the mode and applying one hot encoding
    for col in cat_col:
        df[col].fillna(df[col].mode()[0], inplace=True)

    df = pd.get_dummies(df, columns=cat_col)
    
    # Apply binary encoding for Yes/No values
    for col in bin_col:
        df[col] = df[col].apply(lambda x: 1 if x == "Yes" else 0)

    # Replace the loan approved column with 1 for approved and 0 for denied in the train dataset
    if "Loan_Approved" in df.columns:
        df["Loan_Approved"] = df["Loan_Approved"].apply(lambda x: 1 if x == "Approved" else 0)

    bool_col = df.select_dtypes(include=bool).columns
    df[bool_col] = df[bool_col].astype(int)
    print(df.head())
    print(f"Processed dataset shape: {df.shape}")
    return original_df, df



class XGBoostBiasDetector:
    def __init__(self, **params):
        """
        Class that initializes an XGBoost classifier with customizable hyperparameters.

        This class sets up an XGBoost classifier by allowing to provide optional
        key-value parameters which override the default configurations. Default parameters
        include values for objective function, evaluation metric, maximum tree depth,
        learning rate, number of estimators, random state, and tree method.

        :param params: Dictionary of additional parameters to override default configurations.

        :ivar model: The XGBoost classifier instance initialized with the given configuration.
        :type model: xgb.XGBClassifier
        :ivar feature_names: Stores feature names of the trained model, initially set to None.
        :type feature_names: None or list of str
        :ivar predictions: Stores predictions after the model has been used for predictions, initially set to None.
        :type predictions: None or numpy.ndarray
        """
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 8,
            "learning_rate": 0.01,
            "n_estimators": 200,
            # "subsample": 0.8,
            "random_state": 42,
            "tree_method": "hist"
        }

        # Update the default params with provided parameters
        default_params.update(params)
        self.model = xgb.XGBClassifier(**default_params)
        self.feature_names = None
        self.predictions = None
        
        
    def train(self, x_train, y_train, x_val, y_val):
        """
        Trains the model using the provided training and validation datasets.

        This method trains the model with data provided in `x_train` and `y_train`,
        while also using validation data `x_val` and `y_val` to evaluate the
        model's performance during training. The feature names from the training
        data are saved in `self.feature_names` after training. Outputs progress
        during the training process and confirms the completion when done.

        :param x_train: Training dataset features.
        :param y_train: Training dataset target labels.
        :param x_val: Validation dataset features.
        :param y_val: Validation dataset target labels.
        :return: Returns the current instance of the object after training.
        """
        print("Model is training....")
        self.feature_names = x_train.columns.tolist()
        
        eval_set = [(x_train, y_train), (x_val, y_val)]
        
        self.model.fit(
            x_train, y_train,
            eval_set=eval_set,
            verbose=True
        )
        
        print("Model training completed")
        return self
    
    
    def predict(self, x_test):
        """
        Predicts labels and probabilities for the given input data using the trained model.

        This method uses the model associated with the current object to generate predictions
        and probability estimates for the provided input dataset. Predictions are stored as
        class attribute `self.predictions`, and the probabilities for the positive class
        are stored in `self.probabilities`.

        :param x_test: Input data to be used for predictions. It should match the
                       format and type expected by the model.
        :return: Predicted labels for the input data.
        """
        self.predictions = self.model.predict(x_test)
        return self.predictions
    
    
    # def cross_validate(self, x, y, cv=5):
    #     """
    #     Perform cross-validation
    #     """
    #     scores = cross_val_score(self.model, x, y, cv=cv, scoring="accuracy")
    #     print(f"Cross-validation scores: {scores}")
    #     print(f"Mean CV score: {scores.mean():2f}")
    #     return scores
    
    
    def evaluate_model(self, x_test, y_test):
        """
        Evaluates the performance of the model based on accuracy, precision,
        recall, and F1-score. The predictions are generated for the provided test data,
        and the metrics are calculated against the true labels. Results are returned as a dictionary.

        :param x_test: Test data used for generating predictions.
        :type x_test: Any
        :param y_test: True labels corresponding to the test data.
        :type y_test: Any
        :return: A dictionary containing the evaluation metrics ('accuracy', 'precision',
            'recall', and 'f1') with their calculated values.
        :rtype: dict
        """
        
        predictions = self.predict(x_test)
        
        # Evaluating the model on various evaluation metrics
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions)
        }
        
        print("Model Performance Metrics")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.2f}")
            
        return metrics
    
    
    def anyalyze_bias(self, original_data: pd.DataFrame, predictions, protected_representatives, threshold=0.8):
        """
        Analyzes bias in predictions for specified protected representatives within the dataset.

        This method compares approval statistics for different groups in the dataset, identified
        through `protected_representatives`, in terms of their approval counts, rates, and disparate
        impact. It calculates group-wise statistics and evaluates potential biases by checking
        disparate impact against a threshold which is set at 0.8 by default. Results are returned.

        :param original_data: A pandas DataFrame containing the original dataset with features
            and actual data associated with each entry.
        :param predictions: Predictions generated by the model; these correspond to the
            `original_data`'s entries and are used to compute group-based statistics.
        :param protected_representatives: List of columns in the dataset representing
            protected categories or groups (e.g., gender, ethnicity) based on which bias
            analysis is performed.
        :param threshold: Threshold for disparate impact to be considered a potential bias.

        :return: A dictionary where each key is a protected representative, and its value contains
            approval rates, disparate impact ratio, and detailed group-wise statistics.
        :rtype: dict
        """
        
        bias_results = {}
        
        analysis_data = original_data.iloc[:].copy()
        analysis_data["Predicted_Approval"] = predictions
        
        for repr in protected_representatives:
            print(f"--- Analyzing bias for: {repr} ---")
            
            group_stats = analysis_data.groupby(repr).agg({
                "Predicted_Approval": ["count", "sum", "mean"]
            }).round(3)
            
            group_stats.columns = ["Total_Applications", "Approved_Count", "Approval_Rate"]
            print(group_stats)
            
            approval_rates = analysis_data.groupby(repr)["Predicted_Approval"].mean()
            max_rate = approval_rates.max()
            min_rate = approval_rates.min()
            
            
            disparate_impact = min_rate / max_rate if max_rate > 0 else 0
            
            bias_results[repr] = {
                "approval_rates": approval_rates.to_dict(),
                "disparate_impact": disparate_impact,
                "group_stats": group_stats
            }
            
            print(f"Disparate Impact Ratio: {disparate_impact:.3f}")
            if disparate_impact < threshold:
                print(f"BIAS DETECTED in {repr}- Disparate impact below 0.8 threshold")
                
        return bias_results
    
    
    def create_bias_visualizations(self, original_data: pd.DataFrame, predictions):
        """
        Generates visualizations for analyzing bias in AI model predictions. The method
        produces a set of charts summarizing differences in loan approval rates across
        various demographic groups and highlights potential biases based on approval rate
        disparities.

        :param original_data: A pandas DataFrame containing the original dataset with
            demographic information such as Gender, Race, and Age_Group.
        :param predictions: A list or array-like object containing binary or continuous
            predicted approval rates for each instance in the dataset.
        :return: A matplotlib Figure object containing four subplots visualizing approval
            rates by demographic group and bias analysis.
        """
        
        viz_data = original_data.iloc[:].copy()
        viz_data["Predicted_Approval"] = predictions
        
        plt.style.use('default')
        
        fig, ax = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("AI Bias Detection", fontsize=14, fontweight="bold")
        bias_summary = []
        
        
        # Approval rates by Gender
        gender_approval = viz_data.groupby('Gender')['Predicted_Approval'].mean()
        bias_summary.append(('Gender', gender_approval.max() - gender_approval.min()))
        ax[0, 0].bar(gender_approval.index, gender_approval.values,
                     color=['lightblue', 'lightcoral'], alpha=0.7)
        ax[0, 0].set_title('Loan Approval Rates by Gender', fontweight='bold')
        ax[0, 0].set_ylabel('Approval Rate')
        ax[0, 0].set_ylim(0, 1)
        
        for i, v in enumerate(gender_approval.values):
            ax[0, 0].text(i, v + 0.02, f'{v:2f}', ha='center', fontweight='bold')
        
            
        # Approval rates by Race
        race_approval = viz_data.groupby('Race')['Predicted_Approval'].mean()
        bias_summary.append(('Race', race_approval.max() - race_approval.min()))
        color = plt.cm.Set3(np.linspace(0, 1, len(race_approval)))
        ax[0, 1].bar(race_approval.index, race_approval.values,
                     color=color, alpha=0.7)
        ax[0, 1].set_title('Loan Approval Rates by Race', fontweight='bold')
        ax[0, 1].set_ylabel('Approval Rate')
        ax[0, 1].set_ylim(0, 1)
        ax[0, 1].tick_params(axis='x', rotation=45)
        
        for i, v in enumerate(race_approval.values):
            ax[0, 1].text(i, v + 0.02, f'{v:2f}', ha='center', fontweight='bold')
            
            
        # Approval rates by Age Group
        age_approval = viz_data.groupby('Age_Group')['Predicted_Approval'].mean()
        bias_summary.append(('Age Group', age_approval.max() - age_approval.min()))
        ax[1, 0].bar(age_approval.index, age_approval.values,
                     color='lightgreen', alpha=0.7)
        ax[1, 0].set_title('Loan Approval Rates by Age Group', fontweight='bold')
        ax[1, 0].set_ylabel('Approval Rate')
        ax[1, 0].set_ylim(0, 1)
        ax[1, 0].tick_params(axis='x', rotation=45)
        
        for i, v in enumerate(age_approval.values):
            ax[1, 0].text(i, v + 0.02, f'{v:2f}', ha='center', fontweight='bold')
            
        
        
        attribute, difference = zip(*bias_summary)
        bars = ax[1, 1].bar(attribute, difference, color='orange', alpha=0.7)
        ax[1, 1].set_title('Bias Gap Analysis (Max - Min Approval Rate)', fontweight='bold')
        ax[1, 1].set_ylabel('Approval Rate Difference')
        ax[1, 1].axhline(y=0.1, color='red', linestyle="--", alpha=0.5, label='Potential Bias Threshold')
        ax[1, 1].legend()
        
        for bar, diff in zip(bars, difference):
            ax[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{diff:.3f}', ha="center", fontweight="bold")
            
        
        plt.tight_layout()
        plt.savefig('bias_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    
    def get_feature_importance(self, plot=True, k_feature=15):
        """
        Generates feature importance from the model and optionally visualizes the top features.
        The method retrieves the feature importances from the model, creates a sorted dataframe,
        and displays a horizontal bar plot of the top k features if visualization is enabled.
        The plot is saved as 'feature_importance.png'.

        :param plot: Specifies whether to generate and display the plot for the top k feature
            importances. Defaults to True.
        :type plot: bool
        :param k_feature: Number of top features to display in the plot. Defaults to 15.

        :return: Dataframe containing features and their importance, sorted in descending order
            by importance.
        :rtype: pandas.DataFrame
        """
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        if plot:
            plt.figure(figsize=(10, 8))
            top_features = feature_importance.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 XGBoost Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        return feature_importance
    
    
    def confusion_matrix_analysis(self, y_true, y_pred):
        """
        Analyzes the confusion matrix for a classification task and computes various
        classification metrics. Additionally, it visualizes the confusion matrix
        using a heatmap and saves the plot as an image.

        :param y_true: Ground truth (correct) labels.
        :type y_true: array-like
        :param y_pred: Predicted labels as returned by the classifier.
        :type y_pred: array-like
        :return: A tuple containing a dictionary of computed metrics and the confusion
                 matrix.
        :rtype: tuple
        """
        cm = confusion_matrix(y_true, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'false_positive_rate': fp / (fp + tn),
            'false_negative_rate': fn / (fn + tp)
        }
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Denied', 'Approved'],
                    yticklabels=['Denied', 'Approved'])
        plt.title('Confusion matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        for metric, value in metrics.items():
            print(f'{metric}: {value:.2f}')
        return metrics, cm
        
        
    def generate_shap_analysis(self, x_sample, max_samples=200):
        """
        Generates SHAP analysis for the test sample dataset
        using a Tree-based explainer, and then generates a summary plot of
        SHAP values. The summary plot is saved as an image and displayed.

        :param x_sample: The input dataset to compute SHAP values. Must be a
            DataFrame or dataset compatible with the explainer.
        :param max_samples: The maximum number of data points to sample from the
            dataset for SHAP analysis. Defaults to 100.
        :return: The computed SHAP values for the given dataset sample.
        :rtype: numpy.ndarray
        """
        print("Generating SHAP analysis...")

        x_sample = x_sample.sample(n=max_samples, random_state=42)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(x_sample)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        shap.summary_plot(shap_values, x_sample, max_display=max_samples, show=False)
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()

        return shap_values



def main():
    """
    Main Function to run the model training and evaluation process and also audit for bias.
    """
    print("Starting the model training and evaluation process...")
    filepath = "datasets/loan_access_dataset.csv"
    test_file = "datasets/test.csv"

    original_df, df = preprocessing(filepath)


    X = df.drop('Loan_Approved', axis=1)
    Y = df['Loan_Approved']

    x_train, x_val, y_train, y_val, original_train, original_val = train_test_split(X, Y, original_df, test_size=0.2, random_state=42)
    x_train_split, x_test, y_train_split, y_test, original_train_split, original_test_split = train_test_split(x_train, y_train, original_train, test_size=0.2, random_state=42)


    model = XGBoostBiasDetector()
    model.train(x_train_split, y_train_split, x_val, y_val)

    print("Evaluating model performance on unseen data...")
    metrics = model.evaluate_model(x_test, y_test)

    # Generating predictions for bias analysis
    test_prediction = model.predict(x_test)
    protected_representatives = ["Gender", "Race", "Age_Group", "Employment_Type", "Education_Level", "Citizenship_Status", "Zip_Code_Group"]

    bias_results = model.anyalyze_bias(
        original_test_split,
        test_prediction,
        protected_representatives
    )

    model.create_bias_visualizations(
        original_test_split,
        test_prediction,
    )

    features = model.get_feature_importance()
    print(features)
    matrix, cm = model.confusion_matrix_analysis(y_test, test_prediction)
    print(matrix)

    shap_values = model.generate_shap_analysis(x_train.sample(200, random_state=42))


    test_dataset, processed_test = preprocessing(test_file)

    test_ids = test_dataset['ID']
    # fill missing columns
    missing_cols = set(X.columns) - set(processed_test.columns)
    for col in missing_cols:
        processed_test[col] = 0

    test_predictions = model.predict(processed_test)

    submission = pd.DataFrame({
        'ID': test_ids,
        'Loan_Approved': test_predictions
    })

    submission.to_csv('submission.csv', index=False)


main()