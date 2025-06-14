from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

model = joblib.load("src/models/sentiment_model.pkl")
vectorizer = joblib.load("src/models/vectorizer.pkl")
label_encoder = joblib.load("src/models/label_encoder.pkl")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, review: str = Form(...)):
    vect = vectorizer.transform([review])
    pred = model.predict(vect)[0]
    label = label_encoder.inverse_transform([pred])[0]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": label,
        "review": review
    })
