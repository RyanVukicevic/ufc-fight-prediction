# UFC Fight Prediction Project

## Overview
This project predicts UFC fight outcomes using historical fight data scraped from UFC Stats.

The pipeline:
- Scrapes historical and upcoming fight data
- Cleans and engineers features
- Trains machine learning models (Logistic Regression, Gradient Boosting)
- Performs inference on upcoming fights

## How to Run
pip install -r requirements.txt
python -m src.main

### Install dependencies
pip install -r requirements.txt

## Data
- Data is scraped dynamically from http://ufcstats.com
- No local dataset is required

## Models
- Logistic Regression (with scaling)
- HistGradientBoostingClassifier

## Results
- Accuracy ~60%
- ROC-AUC ~0.65–0.67

## Notes
- Some upcoming fights may not be predicted if fighter history is missing