# Domain-Specific NLP Model

## Overview

This project demonstrates a lightweight NLP text classification pipeline for domain-specific support issues.

## Problem

Generic text workflows do not always classify domain-specific support requests accurately without preprocessing and model training.

## Solution

This project uses TF-IDF features and Logistic Regression to classify text into domain-specific categories such as billing and authentication.

## Workflow

1. Prepare labeled text data
2. Convert text into TF-IDF vectors
3. Train a classification model
4. Evaluate performance
5. Run predictions on new text

## Project Structure

* `data.py` → sample labeled dataset
* `train.py` → model training and evaluation
* `predict.py` → inference on new text

## Example Prediction

Input:
`customer unable to login`

Output:
`authentication`

## Results / Observations

* Demonstrates end-to-end NLP workflow
* Shows preprocessing, model training, and inference
* Provides a clear baseline before upgrading to transformer-based models

## Future Improvements

* Replace TF-IDF with Hugging Face embeddings
* Fine-tune a transformer model
* Add FastAPI inference endpoint
* Expand dataset and evaluation metrics

