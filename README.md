# AI Seller Risk Intelligence

An end-to-end AI and data science system for identifying seller delivery and review risk using temporal machine learning and a FastAPI backend.

## Project Overview

This project analyzes seller behavior over time and predicts two different forms of future seller risk:

- **Delivery risk** — sellers whose future delivery performance may deteriorate.
- **Review risk** — sellers whose future customer review performance may deteriorate.

The system combines:

- temporal feature engineering
- time-aware model validation
- Random Forest and Logistic Regression
- model comparison and selection
- error analysis
- calibration analysis
- temporal drift analysis
- SHAP explainability
- FastAPI model serving
- automated API testing

The goal is not simply to train a classifier, but to build a complete machine-learning system that can move from historical data to an operational prediction API.

---

## Business Problem

Seller performance can deteriorate gradually before the problem becomes obvious.

A seller may begin experiencing:

- longer delivery times
- increasing delivery delays
- higher late-delivery rates
- declining review scores
- increasing low-rated reviews

A useful risk system should identify these signals early enough for an operations team to investigate.

This project therefore treats seller risk as an **early-warning and prioritization problem**, rather than as an automatic decision-making system.

---

## Solution

The system uses historical seller behavior to construct temporal snapshots.

Each seller snapshot contains information available up to that point in time.

The models then use this historical information to estimate future delivery or review risk over a later prediction window.

The final architecture contains two target-specific models:

### Delivery Risk Model

**Model:** Random Forest

**Role:** Monitoring / early warning

**Threshold:** 0.15

**Out-of-time ROC-AUC:** approximately 0.51

The delivery model showed weak out-of-time discrimination and should therefore be treated as a monitoring signal rather than a reliable standalone classifier.

### Review Risk Model

**Model:** Logistic Regression

**Role:** Screening / prioritization

**Threshold:** 0.12

**Out-of-time ROC-AUC:** approximately 0.66

The review model provides stronger out-of-time ranking performance, but its precision remains limited. It is therefore intended to support prioritization and human review rather than make automatic decisions.

---

## Machine Learning Approach

The project uses a temporal evaluation strategy to reduce leakage and better represent real-world deployment.

### Data flow

```text
Raw Olist data
      ↓
Data cleaning
      ↓
Seller feature engineering
      ↓
Temporal seller snapshots
      ↓
Target-specific datasets
      ↓
Time-based train / validation / test
      ↓
Model training
      ↓
Validation and threshold selection
      ↓
Untouched temporal test set
      ↓
Final model selection