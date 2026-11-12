# 06. Prediction System

## Purpose
This folder encapsulates the production inference pipeline that accepts new hyperspectral measurements and generates real-time disease diagnostic predictions.

## Contents
- `predict.py`: Standalone inference service that loads trained model pipelines from `05_Machine_Learning/models/`, preprocesses incoming spectral vectors, extracts indices, and outputs disease stage probabilities alongside agronomic recommendations.

