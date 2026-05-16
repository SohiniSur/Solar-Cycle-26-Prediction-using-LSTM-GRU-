# Solar Cycle 26 Peak Prediction using LSTM / GRU Ensemble

> Forecasting the amplitude and timing of Solar Cycle 26 using deep learning and ensemble stacking on 276 years of sunspot data.

**Research Internship | IIIT Kalyani | May 2025 – Jul 2025**

---

## Problem Statement

Solar cycle prediction is a critical open problem in space weather forecasting. Accurate peak-amplitude forecasts for Solar Cycle 26 inform satellite operations, power grid management, and radio communications planning. Traditional statistical methods fail to capture the long-term temporal dependencies in solar activity data.

---

## Dataset

| Property | Detail |
|---|---|
| Source | NASA / WDC-SISLO (World Data Center for the Sunspot Index and Long-term Solar Observations) |
| Coverage | 1749 – 2025 |
| Observations | 3,316 monthly mean sunspot number records |
| Target | Monthly smoothed sunspot number (SSN) |

---

## Architecture

A **3-stage ensemble pipeline**:

```
Raw Sunspot Data (3,316 obs)
        │
        ▼
  Sequence Windows (look-back: configurable)
        │
   ┌────┴────┐
   │         │
 LSTM       GRU          ← Stage 1: Base Learners
   │         │
   └────┬────┘
        │  (concatenated outputs)
        ▼
   XGBoost Meta-Learner  ← Stage 2: Stacking
        │
        ▼
  132-month Forecast     ← Solar Cycle 26 peak amplitude & timing
```

**Stage 1 — Base Learners:**
- Stacked LSTM network (TensorFlow/Keras) capturing long-range dependencies
- Stacked GRU network as complementary learner with different inductive bias

**Stage 2 — Meta-Learner:**
- XGBoost regressor trained on out-of-fold LSTM and GRU predictions
- Synthesizes both neural network outputs into a final forecast

---

## Results

| Model | Test RMSE | Test MAE |
|---|---|---|
| Linear Regression (baseline) | 70.85 | — |
| LSTM (standalone) | 19.09 | 13.37 |
| GRU (standalone) | 19.13 | 13.74 |
| **XGBoost Ensemble (LSTM + GRU)** | **17.26** | **11.50** |

**The ensemble achieves ~73% improvement over the linear baseline** and outperforms both standalone deep learning models on Test RMSE and MAE.

---

## Key Findings

- GRU and LSTM produce statistically similar outputs on this dataset; stacking them captures complementary residuals that neither model alone corrects.
- XGBoost meta-learning provides measurable gains over simple averaging, indicating non-linear interactions between base model errors.
- 132-month multi-output forecast generated for Solar Cycle 26, with quantified uncertainty across the prediction horizon.

---

## Tech Stack

| Tool | Usage |
|---|---|
| Python | Core implementation |
| TensorFlow / Keras | LSTM and GRU architectures |
| XGBoost | Meta-learner stacking |
| pandas / NumPy | Data preprocessing and sequence construction |
| matplotlib / seaborn | Visualization and cycle plots |


---

## References

- WDC-SISLO Sunspot Data: https://www.sidc.be/SILSO/
- NASA Solar Cycle Progression: https://www.swpc.noaa.gov/products/solar-cycle-progression

---

*Part of research internship at IIIT Kalyani, May–Jul 2025.*
