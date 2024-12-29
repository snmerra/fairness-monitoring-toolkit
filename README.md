# Fairness Monitoring Toolkit

This is a learning toolkit for auditing and monitoring fairness in machine learning data models.
This project evaluates model performance using fairness metrics such as True Positive Rate (TPR), False Positive Rate (FPR), Precision, and Disparate Impact, while supporting group and intersectional analyses for sensitive attributes like gender and race.

---

## Key Features

1. **Fairness Auditing**:
   - Evaluate overall metrics (Normal Metrics) for model fairness.
   - Group-specific fairness checks for sensitive attributes (e.g., gender, race).
   - Classification of metrics as **Good**, **Fair**, or **Not Fair** based on industry standards like the Four-Fifths Rule.

2. **Synthetic Data Generation**:
   - Generate synthetic datasets to simulate fairness scenarios in a controlled, privacy-safe environment.

3. **Visualizations**:
   - Generate bar charts and disparity plots to highlight fairness results across groups.

4. **Interpretable Metrics**:
   - Built-in thresholds and explanations for metrics to guide fairness assessments.

5. **Extensibility**:
   - Designed for integration with Continuous Fairness Monitoring pipelines.

---
## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- Git

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/fairness-monitoring-toolkit.git
   cd fairness-monitoring-toolkit
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the toolkit:
   ```bash
   python3 fairness_audit/fairness_analysis.py
   ```

---

## Usage

### **1. Generate Synthetic Data**

Generate a synthetic dataset for testing fairness metrics:
```bash
python3 fairness_audit/generate_synthetic_data.py
```
The dataset will be saved in the `data/` folder as `synthetic_data.csv`.

### **2. Perform Fairness Auditing**

Run the fairness auditing script:
```bash
python3 fairness_audit/run_aequitas.py
```

### **3. Metrics Evaluated**

The toolkit evaluates the following metrics:

| Metric                         | Description                                                                 | Threshold/Benchmark      |
|--------------------------------|-----------------------------------------------------------------------------|--------------------------|
| **True Positive Rate (TPR)**   | Proportion of actual positives correctly predicted.                         | Fair: ≥ 0.8, Good: ≥ 0.85|
| **False Positive Rate (FPR)**  | Proportion of actual negatives incorrectly predicted as positives.          | Fair: ≤ 0.2, Good: ≤ 0.15|
| **Precision**                  | Proportion of predicted positives that are actually positives.              | Fair: ≥ 0.7, Good: ≥ 0.85|
| **Disparate Impact (DI)**      | Ratio of selection rates between groups (e.g., gender, race).               | Fair: 0.8–1.25           |
| **Positive Prevalence Rate**   | Proportion of positives in the dataset.                                    | Monitored for balance.   |

The metrics are classified as:
- **Good**: Metric is optimal and aligns with fairness benchmarks.
- **Fair**: Metric is acceptable but requires improvement.
- **Not Fair**: Metric shows potential bias and needs immediate attention.

---

## Visualizations

The toolkit generates visualizations for group metrics (e.g., TPR, FPR, Precision) by attributes like gender and race:
```python
# Example visualization in run_aequitas.py
plot_group_metrics(metrics_df, attribute="gender")
```

Example visualization:

---

## Ideal Audience

- **Data Scientists**: To evaluate fairness in machine learning models.
- **Software Testers**: To validate fairness compliance in systems.
- **Researchers**: To test fairness metrics on synthetic and real-world data.

---

## Future Enhancements

- Continuous Fairness Monitoring: Automate fairness checks via CI/CD pipelines like GitHub Actions.
- Integration with Explainability Tools: Incorporate SHAP/LIME for insights into model decision-making.
- Real-World Dataset Support: Extend support for real-world data.

---

## Contributing

We welcome contributions! Please fork the repository and submit a pull request. Ensure your code adheres to the project style and includes documentation.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions or feedback, please open an issue or reach out via [GitHub Discussions](https://github.com/<username>/fairness-monitoring-toolkit/discussions).

