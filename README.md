# 📝 Product Review Sentiment Analyzer

A lightweight machine learning-powered web application to analyze sentiment in product reviews.  
Simply enter a review, and the app will classify it as **Positive**, **Neutral**, or **Negative**.

---

## 🚀 Features

- Balances Amazon review data (thankfully, I found a cleaned Amazon Reviews dataset 👉 [Kaggle Dataset](https://www.kaggle.com/datasets/danielihenacho/amazon-reviews-dataset))
- Trains and evaluates multiple models (Logistic Regression, SVM, Naive Bayes)
- Selects the best-performing model via GridSearchCV
- Predicts sentiment from user input using the trained model
- User-friendly web interface built with FastAPI and Jinja2

---

## 🛠 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Prerana-G/product-review-sentiment-analysis.git
cd product-review-sentiment-analysis
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv

# For Windows
venv\Scripts\activate

# For macOS/Linux
source venv/bin/activate
```

### 3. Install the Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
uvicorn app.main:app --reload
```

Then open your browser and go to 👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)

Type a product review like:

> “This product was AMAZING and changed my life!”

...and the app will display the predicted sentiment 🎯

---

## 🧠 Tech Stack

- **Python 3.10+**
- **pandas**, **scikit-learn**, **joblib** – for data processing and machine learning
- **FastAPI** – for building the backend API
- **Jinja2**, **HTML**, **CSS** – for the frontend UI
- **Uvicorn** – ASGI server to run the FastAPI app

---

## 🚀 Live Demo  
Check out the live sentiment analysis app here:  
👉 [**Sentilyzer on Render**](https://sentiment-analyzer-2t9h.onrender.com/)

---

## 💌 Thanks for Visiting!

This was a fun project.  
Hope you enjoy using it as much as I enjoyed building it! 😄

Feel free to⭐ Star the repo, 🍴 Fork it, 💬 Drop a suggestion or issue
