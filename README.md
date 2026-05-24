# Phishing Detection

AI Phishing Shield is a Python project for detecting phishing attempts across URLs, emails, and SMS messages.
The application combines machine learning, natural language processing, and heuristic feature extraction to classify suspicious content and explain why it may be dangerous.

## Features

- URL phishing detection using a trained XGBoost model
- SMS phishing detection with DistilBERT + XGBoost and BiLSTM fallback models
- Email phishing detection using keyword, structural, and semantic features
- Streamlit web app interface for easy testing and reporting
- Explainable phishing categories such as urgency, credential requests, threats, and impersonation

## Repository Structure

- `app.py` - Streamlit app and prediction workflow
- `train_email_bilstm.py` - Train email BiLSTM model
- `train_email_distilbert_xgb.py` - Train DistilBERT + XGBoost email model
- `train_sms_bilstm.py` - Train SMS BiLSTM model
- `train_sms_distilbert_xgb.py` - Train DistilBERT + XGBoost SMS model
- `train_url_xgboost.py` - Train URL XGBoost model
- `test_models.py` - Model evaluation and testing utilities
- `dataset/` - Raw datasets used for training and experimentation
- `email_xgb_model/`, `sms_xgb_model/`, `sms_model_bilstm/` - Model artifacts and metadata for each classifier

## Requirements

Install dependencies using pip:

```bash
pip install streamlit torch transformers xgboost scikit-learn pandas numpy joblib plotly scipy
```

## Run the App

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Then open the displayed local address in your browser.

## Training Models

You can retrain the project models using the provided scripts. Example:

```bash
python train_url_xgboost.py
python train_email_distilbert_xgb.py
python train_sms_distilbert_xgb.py
```

## Notes

Large binary model files are intentionally excluded from the Git repository via `.gitignore`.
If you need to run the app from a fresh clone, train the models locally or add the required model artifacts manually.

## License

This repository does not include a license file. Add a license if you want to share the project publicly.
