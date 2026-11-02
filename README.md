# ML Drift Monitoring Lab

**Understanding Data Drift, Concept Drift, and Model Degradation in Machine Learning**

An end-to-end, hands-on ML monitoring project exploring how data distributions change, why predictive models degrade, and how statistical drift detectors behave in practice.

Built using Python, Scikit-learn, SciPy, River, Plotly, and Streamlit.

## Overview

Machine learning models can deteriorate as incoming data and the relationships underlying predictions evolve.

But an important question is:

**Does detecting data drift actually tell us when model performance is degrading?**

This project investigates that question using controlled synthetic experiments and the real-world Electricity (ELEC2) dataset.

## What I Built

- Synthetic simulations of feature drift and concept drift.
- KS-test, PSI, ADWIN, and Page-Hinkley drift detection.
- Repeated experiments measuring false-alert rates and detection delays.
- Real-world electricity-market model monitoring.
- Feature-level drift analysis and model coefficient comparisons.
- Chronological model validation using lagged features.
- An interactive Streamlit dashboard for exploring the results.

## Key Findings

### 1. Drift detection does not guarantee model degradation

On the ELEC2 dataset, PSI for NSW electricity demand had almost no monotonic association with same-batch model accuracy:

**Spearman correlation: -0.0115**

Large changes in this feature did not consistently coincide with lower accuracy.

### 2. Repeated statistical tests can produce misleading alarms

In the synthetic concept-drift experiment, the KS test raised at least one alarm in 80% of runs within the evaluation window.

However, it also alerted in 80% of equivalent no-drift runs.

This demonstrates why monitoring systems need no-drift controls and careful alert policies.

### 3. Different detectors identify different changes

In our initial ten-seed benchmark:

| Detector | Feature-drift alert probability | Concept-drift alert probability |
|---|---:|---:|
| KS | 100% | 80% |
| PSI | 100% | 0% |
| ADWIN | 100% | 0% |
| Page-Hinkley | 0% | 100% |

These are probabilities of at least one alert within the selected windows, not general-purpose detection guarantees. Page-Hinkley used delayed ground-truth labels, whereas the feature detectors operated without labels.

### 4. Feature drift and model importance are different

Three electricity-market features were constant during the initial training period but varied in later observations.

Although some of these features showed substantial distribution changes, adding them to the later-trained logistic regression model did not materially improve test performance.

### 5. Recent training data improved performance in this experiment

Using lagged features and a shared chronological evaluation period:

| Metric | Early model | Recent model |
|---|---:|---:|
| Accuracy | 0.6675 | 0.6937 |
| ROC-AUC | 0.7362 | 0.7448 |
| Log loss | 0.7115 | 0.6221 |

The more recently trained model performed better on this test period. This does not isolate the causal effect of retraining, since the two training windows differ.

## Interactive Dashboard

The Streamlit dashboard includes:

- Model performance trends over time.
- Adjustable PSI alert thresholds.
- Drift detector comparisons.
- Temporal model comparison and rolling accuracy charts.

**Live demo:** Link to be added after deployment.

## Project Structure

```text
drift-research/
├── app.py
├── notebooks/
│   ├── 01_understanding_drift.ipynb
│   ├── 02_monitoring_model_degradation.ipynb
│   ├── 03_comparing_drift_detectors.ipynb
│   ├── 04_drift_detector_reliability.ipynb
│   ├── 05_real_world_drift.ipynb
│   ├── 06_feature_drift_vs_model_impact.ipynb
│   └── 07_temporal_model_validation.ipynb
├── results/
├── requirements.txt
└── README.md
```

## Run Locally

Clone the repository and open it in a terminal.

Create a Python 3.12 virtual environment and install dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Launch the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard reads previously generated CSV artifacts from the `results/` directory.

## Limitations

- Synthetic results use a limited number of simulation seeds and selected detector thresholds.
- Real-world concept-drift dates are not independently labeled.
- Repeated statistical tests can generate chance alerts.
- The Electricity dataset is a historical benchmark; feature timing and availability are not fully verified for a real-time forecasting application.
- Correlations between drift metrics and model performance do not establish causality.
- The dashboard is an exploratory monitoring demonstration, not a production alerting or automatic retraining system.

## Dataset and Tools

**Dataset:** Electricity (ELEC2), accessed through the River Python library.

**Libraries:** NumPy, Pandas, Scikit-learn, SciPy, River, Matplotlib, Plotly, Streamlit.

## Conclusion

The main lesson from this project is that **detecting distribution change and identifying actionable model degradation are different problems**.

Useful ML monitoring requires appropriate baselines, understanding of label availability, careful alert evaluation, and evidence that an observed change matters for predictive performance.