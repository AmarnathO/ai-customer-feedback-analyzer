import spaces
import gradio as gr
from transformers import pipeline
from functools import lru_cache

# Load model only when first needed
@lru_cache(maxsize=1)
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

# ZeroGPU requires a decorated function
@spaces.GPU(duration=60)
def analyze_sentiment(review):
    if not review.strip():
        return "Please enter a customer review."

    classifier = load_model()
    result = classifier(review, truncation=True)[0]

    label = result["label"]
    score = result["score"]

    return f"Sentiment: {label}\nScore: {score:.2%}"

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        label="Customer Review",
        lines=3
    ),
    outputs=gr.Textbox(label="AI Prediction"),
    title="AI Customer Feedback Analyzer"
)

demo.launch()
