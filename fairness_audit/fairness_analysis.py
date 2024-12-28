import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


# Load the dataset
def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Parameters:
        file_path (str): Path to the dataset.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


# Compute Normal Metrics (Overall Fairness Check)
def compute_normal_metrics(df):
    """
    Compute overall metrics for the model.

    Parameters:
        df (pd.DataFrame): Input data.

    Returns:
        None
    """
    try:
        y_true = df['label_value']
        y_pred = df['model_prediction']

        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total = tn + fp + fn + tp  # Total number of samples

        # Ratios for each confusion matrix element
        tn_ratio = tn / total
        fp_ratio = fp / total
        fn_ratio = fn / total
        tp_ratio = tp / total

        print("\n=== Normal Metrics: Overall Fairness Check ===")
        print(f"Confusion Matrix (in ratios):")
        print(f"True Negative: {tn_ratio:.2f}, False Positive: {fp_ratio:.2f}, False Negative: {fn_ratio:.2f}, True Positive: {tp_ratio:.2f}")

        # Overall Metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        positive_rate = y_pred.mean()

        print(f"Accuracy: {accuracy:.2f} ({evaluate_classification(accuracy, 0.7, 0.85)})")
        print(f"F1 Score: {f1:.2f} ({evaluate_classification(f1, 0.7, 0.85)})")
        print(f"Statistical Parity (Overall Positive Rate): {positive_rate:.2f} ({evaluate_classification(positive_rate, 0.4, 0.6)})")

        # Explanation of classifications
        print("\nExplanation of Evaluations:")
        print("- Good: Metric is strong, aligns with fairness benchmarks (e.g., Accuracy ≥ 0.85).")
        print("- Fair: Metric is acceptable but has room for improvement (e.g., 0.7 ≤ Accuracy < 0.85).")
        print("- Not Fair: Metric indicates potential bias (e.g., Accuracy < 0.7).")

    except Exception as e:
        print(f"Error in compute_normal_metrics: {e}")
        raise


# Evaluate Metric Classification (Based on Standards)
def evaluate_classification(value, fair_lower, good_lower):
    """
    Evaluate metric value as Good, Fair, or Not Fair based on standardized thresholds.

    Parameters:
        value (float): Metric value to evaluate.
        fair_lower (float): Lower threshold for Fair classification.
        good_lower (float): Lower threshold for Good classification.

    Returns:
        str: Evaluation result (Good/Fair/Not Fair).
    """
    if value >= good_lower:
        return "Good"
    elif value >= fair_lower:
        return "Fair"
    else:
        return "Not Fair"


# Compute Group Metrics (Fairness Check by Group)
def compute_group_metrics(df, attribute):
    """
    Compute fairness metrics for specific groups.

    Parameters:
        df (pd.DataFrame): Input data.
        attribute (str): Sensitive attribute to evaluate.

    Returns:
        pd.DataFrame: Group metrics.
    """
    try:
        print(f"\n=== Group Metrics: Fairness Check by {attribute} ===")

        groups = df[attribute].unique()
        metrics = []

        for group in groups:
            group_data = df[df[attribute] == group]
            group_y_true = group_data['label_value']
            group_y_pred = group_data['model_prediction']

            # Compute Group Metrics
            tpr = sum((group_y_pred == 1) & (group_y_true == 1)) / sum(group_y_true == 1) if sum(group_y_true == 1) > 0 else 0
            fpr = sum((group_y_pred == 1) & (group_y_true == 0)) / sum(group_y_true == 0) if sum(group_y_true == 0) > 0 else 0
            precision = sum((group_y_pred == 1) & (group_y_true == 1)) / sum(group_y_pred == 1) if sum(group_y_pred == 1) > 0 else 0
            prevalence = sum(group_y_true == 1) / len(group_y_true)

            metrics.append({
                "Group": group,
                "TPR": tpr,
                "TPR Evaluation": evaluate_classification(tpr, 0.8, 0.85),
                "FPR": fpr,
                "FPR Evaluation": evaluate_classification(1 - fpr, 0.8, 0.85),
                "Precision": precision,
                "Precision Evaluation": evaluate_classification(precision, 0.7, 0.85),
                "Prevalence": prevalence,
                "Prevalence Evaluation": evaluate_classification(prevalence, 0.4, 0.6),
            })

            print(f"\nGroup: {group}")
            print(f"True Positive Rate (TPR): {tpr:.2f} ({evaluate_classification(tpr, 0.8, 0.85)})")
            print(f"False Positive Rate (FPR): {fpr:.2f} ({evaluate_classification(1 - fpr, 0.8, 0.85)})")
            print(f"Precision: {precision:.2f} ({evaluate_classification(precision, 0.7, 0.85)})")
            print(f"Positive Prevalence Rate: {prevalence:.2f} ({evaluate_classification(prevalence, 0.4, 0.6)})")

        # Explanation of classifications
        print("\nExplanation of Evaluations:")
        print("- Good: Metric is close to parity or optimal thresholds (e.g., TPR ≥ 0.85).")
        print("- Fair: Metric is within acceptable ranges (e.g., 0.8 ≤ TPR < 0.85).")
        print("- Not Fair: Metric shows significant group disparity (e.g., TPR < 0.8).")

        return pd.DataFrame(metrics)

    except Exception as e:
        print(f"Error in compute_group_metrics: {e}")
        raise


# Plot Group Metrics
def plot_group_metrics(metrics_df, attribute):
    """
    Plot group metrics for visualization.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame of group metrics.
        attribute (str): Attribute used for grouping.

    Returns:
        None
    """
    try:
        metrics_df.set_index("Group")[["TPR", "FPR", "Precision"]].plot(kind="bar", figsize=(10, 6))
        plt.title(f"Fairness Metrics by {attribute}")
        plt.xlabel(attribute)
        plt.ylabel("Metric Value")
        plt.legend(title="Metrics")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plot_group_metrics: {e}")
        raise


# Data Interpretation
def interpret_metrics():
    """
    Provide insights and interpretation of computed metrics.

    Returns:
        None
    """
    print("\n=== Data Interpretation ===")
    print("1. Normal metrics (Overall Fairness Check) give an overview of the model's performance.")
    print("2. Group metrics (Fairness Check by Group) highlight disparities for sensitive attributes (e.g., gender, race).")
    print("3. Evaluations classify metrics as Good, Fair, or Not Fair, based on widely accepted standards.")


if __name__ == "__main__":
    # Path to the dataset
    data_path = "data/synthetic_data.csv"

    # Load data
    df = load_data(data_path)

    # Compute Normal Metrics
    compute_normal_metrics(df)

    # Compute Group Metrics for gender and race
    attributes_to_evaluate = ['gender', 'race']
    for attr in attributes_to_evaluate:
        group_metrics_df = compute_group_metrics(df, attr)
        plot_group_metrics(group_metrics_df, attr)

    # Interpret Metrics
    interpret_metrics()
