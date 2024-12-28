import pandas as pd
import numpy as np

def generate_synthetic_data(output_path="data/synthetic_data.csv", num_samples=1000):
    """
    Generate synthetic data for fairness analysis with realistic credit scores.

    Parameters:
        output_path (str): Path to save the synthetic dataset.
        num_samples (int): Number of samples to generate.

    Returns:
        None
    """
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic data
    data = {
        'gender': np.random.choice(['Male', 'Female'], size=num_samples),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], size=num_samples),
        'income': np.random.randint(20000, 100000, size=num_samples),  # Annual income
        'education': np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], size=num_samples),
        'label_value': np.random.choice([0, 1], size=num_samples),  # Ground truth labels
        'model_prediction': np.random.choice([0, 1], size=num_samples),  # Model predictions
        'score': np.random.randint(300, 851, size=num_samples)  # Credit scores (300-850)
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Synthetic data with credit scores saved to {output_path}")


if __name__ == "__main__":
    # Path to save the dataset
    output_file = "data/synthetic_data.csv"

    # Generate data
    generate_synthetic_data(output_path=output_file, num_samples=1000)
