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


classifier = load_model()

# ZeroGPU requires a decorated function
@spaces.GPU(duration=60)
def analyze_reviews(text):

    # Convert multiline text into a list of reviews
    reviews = [
        review.strip()
        for review in text.split("\n")
        if review.strip()
    ]

    if not reviews:
        return []

    # Run batch inference
    results = classifier(
        reviews,
        batch_size=4,
        truncation=True
    )

    # Prepare rows for Gradio
    output = []

    for review, result in zip(reviews, results):
        output.append([
            review,
            result["label"],
            round(result["score"], 3)
        ])

    return output
demo = gr.Interface(
    fn=analyze_reviews,
    inputs=gr.Textbox(
        label="Customer Reviews",
        lines=6,
        placeholder="Enter one review per line"
    ),
    outputs=gr.Dataframe(
        headers=["Customer Review", "Sentiment", "Score"],
        datatype=["str", "str", "number"]
    ),
    title="AI Customer Feedback Analyzer",
    description="Analyze multiple customer reviews using Hugging Face."
)

demo.launch()
