````md id="m7k2pa"
# Aircraft Engine Predictive Maintenance with RNN & LSTM (NASA C-MAPSS)

## Overview

This repository contains a Jupyter Notebook implementation for **predictive maintenance** using deep learning on the **NASA C-MAPSS turbofan engine dataset**.

The project focuses on predicting whether an engine is close to failure using historical sensor signals and operating conditions.

The notebook demonstrates how to build and compare:

- Simple RNN
- Stacked RNN
- LSTM Networks

for **binary engine failure prediction**.

---

## Problem Statement

Aircraft engines gradually degrade over time.

The goal is:

> Given past sensor readings, predict if the engine will fail within the next fixed number of cycles.

This helps airlines perform maintenance **before breakdown happens**.

Why? Because early detection saves:

- Money
- Downtime
- Repair costs
- Safety risks

---

## Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**

The dataset contains multiple simulated aircraft engines run until failure.

Each engine includes:

- Engine ID
- Cycle number
- 3 operational settings
- 21 sensor measurements

---

## Target Creation

This notebook converts Remaining Useful Life (RUL) into a binary label.

Example:

```text id="1d2fga"
If RUL <= 30 cycles  -> 1 (Failure Soon)
If RUL > 30 cycles   -> 0 (Healthy)
````

This turns the task into a classification problem.

---

## Workflow

## 1. Data Loading

Training, testing, and truth datasets are loaded:

```python id="8m2lqe"
PM_train.txt
PM_test.txt
PM_truth.txt
```

---

## 2. Data Preprocessing

Includes:

* Removing empty columns
* Naming features
* Sorting by engine ID and cycle
* Calculating RUL
* Normalization / scaling

---

## 3. Sequence Generation

Sliding windows are created from time-series data.

Example:

```text id="q1m6sw"
Past 50 cycles -> Predict failure risk
```

Why? Because neural networks need sequences.

---

## 4. Deep Learning Models

### Vanilla RNN

Basic recurrent memory model.

### Stacked RNN

Multiple recurrent layers for stronger pattern learning.

### LSTM

Uses gates to remember useful long-term information.

Aha! This is why programmers love LSTM.

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## Run Notebook

Install dependencies:

```bash id="ovv4is"
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn jupyter
```

Launch notebook:

```bash id="g6ndx1"
jupyter notebook
```

Open:

```text id="j5blfi"
mmain.ipynb
```

---

## Expected Outputs

* Model training accuracy
* Validation loss
* Confusion matrix
* Precision / Recall
* Failure classification results

---

## Repository Structure

```text id="v44g3q"
├── mmain.ipynb
├── data/
│   ├── PM_train.txt
│   ├── PM_test.txt
│   └── PM_truth.txt
├── README.md
└── requirements.txt
```

---

## Example Insight

If sensor temperature, vibration, or pressure starts drifting over many cycles:

* RNN may forget older signals
* LSTM can remember long degradation trends

That is why LSTM often performs better.

---

## Future Improvements

* Bidirectional LSTM
* Attention mechanism
* Transformer models
* Remaining Useful Life regression
* Explainable AI (SHAP)

---

## Author

Youssef Aittizi

---

