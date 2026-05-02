````md
# Aircraft Engine Failure Prediction using RNN, BiRNN, and LSTM (NASA C-MAPSS)

## Overview

This repository contains the complete implementation of a comparative deep learning study for **early aircraft engine failure prediction** using recurrent neural networks on the **NASA C-MAPSS** benchmark dataset.

The objective is to detect whether an aircraft engine will enter a **failure-critical state** within a predefined future warning horizon using historical multivariate sensor data.

This project compares four sequence learning architectures:

- Single-Feature Vanilla RNN  
- Multi-Feature Stacked RNN  
- Multi-Feature Bidirectional RNN (BiRNN)  
- Stacked Long Short-Term Memory (LSTM)

The study shows that **LSTM achieved the best predictive performance**, confirming its strength in learning long-term degradation behavior from time-series sensor data.

---

## Dataset

**Dataset:** NASA Commercial Modular Aero-Propulsion System Simulation (**C-MAPSS**)

The dataset contains simulated run-to-failure trajectories of turbofan engines under different operating conditions and fault modes.

Each engine unit includes:

- 3 operational setting variables
- 21 sensor measurements
- cycle-by-cycle degradation progression

---

## Problem Formulation

We formulate the task as a **binary classification problem**:

Given the last `Ts = 50` operating cycles, predict whether the engine will fail within the next:

```text
w1 = 30 cycles
````

Output label:

* `1` = Failure approaching soon
* `0` = Normal condition

---

## Models Compared

### 1. Single-Feature Vanilla RNN

Uses one selected sensor only.

### 2. Multi-Feature Stacked RNN

Uses all sensor channels with deeper recurrent layers.

### 3. Bidirectional RNN (BiRNN)

Learns forward and backward temporal dependencies.

### 4. Stacked LSTM

Uses gating mechanisms to retain useful long-term information.

---

## Final Results

| Model      | Accuracy  | Precision | Recall    | F1 Score  |
| ---------- | --------- | --------- | --------- | --------- |
| Single RNN | 72.0%     | --        | --        | 51.1%     |
| Multi RNN  | 82.0%     | --        | --        | 67.3%     |
| BiRNN      | 84.9%     | --        | --        | 81.2%     |
| LSTM       | **93.7%** | **94.4%** | **87.0%** | **90.6%** |

---

## Key Finding

LSTM significantly outperformed standard RNN models.

Why? Because:

* Vanilla RNN struggles with long memory
* Engine degradation happens slowly over many cycles
* LSTM remembers useful signals and forgets noise

Aha! This is why programmers love LSTM for time-series forecasting.

---

## Repository Structure

```text
├── data/
│   └── CMAPSSData/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── preprocess.py
│   ├── windows.py
│   ├── train.py
│   ├── evaluate.py
│   └── models/
│       ├── rnn.py
│       ├── birnn.py
│       └── lstm.py
├── results/
│   ├── metrics.csv
│   └── plots/
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yourusername/engine-failure-rnn.git
cd engine-failure-rnn
pip install -r requirements.txt
```

---

## Run Training

```bash
python src/train.py --model lstm
```

Other choices:

```bash
--model rnn
--model stacked_rnn
--model birnn
```

---

## Example (PyTorch)

```python
model = LSTMClassifier(
    input_size=24,
    hidden_size=64,
    num_layers=2
)
```

---

## Visual Outputs

* Training vs validation loss
* ROC curve
* Precision-recall curve
* Confusion matrix
* Remaining Useful Life trend plots

---

## Applications

* Predictive maintenance
* Fleet health monitoring
* Aircraft safety systems
* Industrial rotating machines
* Turbine monitoring

---

## Future Improvements

* Attention-LSTM
* Transformer models
* Remaining Useful Life regression
* Explainable AI (SHAP / GradCAM)
* Real-time deployment

---

## Citation

If you use this repository, please cite:

```bibtex
@article{engine_rnn_failure,
  title={Early Aircraft Engine Failure Prediction using Recurrent Neural Networks},
  author={Your Name},
  year={2026}
}
```

---

## Contact

For collaboration or research discussion:

GitHub: `yourusername`

---

## License

MIT License

```
```
