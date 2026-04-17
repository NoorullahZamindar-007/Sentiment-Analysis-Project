# Sentiment Analysis Project

A simple Flask web app for sentiment analysis using a TF-IDF vectorizer and a Logistic Regression model trained on the NLTK `movie_reviews` dataset.

## Features

- Flask backend with HTML/CSS frontend
- Text preprocessing that matches the training pipeline
- Positive/negative sentiment prediction
- Confidence score display when available
- Empty input handling
- Training script to regenerate model files

## Project Structure

```text
Sentiment Analysis Project/
├── app.py
├── train.py
├── requirements.txt
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Requirements

- Python 3.12 recommended
- Windows PowerShell or terminal

## Setup

Create a virtual environment:

```powershell
py -3.12 -m venv .venv
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the Model

If the `.pkl` files are missing or invalid, regenerate them with:

```powershell
.\.venv\Scripts\python.exe train.py
```

This will:

- download required NLTK data
- clean the dataset text
- train the TF-IDF vectorizer
- train the Logistic Regression model
- save `sentiment_model.pkl`
- save `tfidf_vectorizer.pkl`

## Run the App

Start the Flask server:

```powershell
.\.venv\Scripts\python.exe app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## How It Works

The app preprocesses input text using the same steps used during training:

1. Convert text to lowercase
2. Remove punctuation and digits
3. Remove extra spaces
4. Remove English stopwords
5. Transform text with TF-IDF
6. Predict sentiment with Logistic Regression

## Example

Input:

```text
I love this movie. It was amazing.
```

Output:

```text
Positive
```

## Troubleshooting

### PowerShell cannot run `Activate.ps1`

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### The app says the vectorizer is not fitted

Run:

```powershell
.\.venv\Scripts\python.exe train.py
```

Then restart the Flask app.

### Wrong Python version is being used

Check installed Python versions:

```powershell
py -0p
```

Use Python 3.12 when creating the virtual environment.

## Notes

- The model is trained on the NLTK `movie_reviews` dataset, so very short custom phrases may not always produce human-intuitive results.
- The app is intended as a learning/demo project and can be extended with your own dataset later.
