# AI Customer Feedback Analyzer

An AI-powered web application that analyzes customer feedback using a pretrained Hugging Face sentiment classification model.

## Business Problem

Product and customer experience teams receive large amounts of unstructured feedback. Manually reviewing individual comments can be time-consuming.

This project demonstrates how machine learning can help classify customer sentiment and provide an initial signal for feedback analysis.

## Current Features

- Analyze individual customer reviews.
- Classify reviews as POSITIVE or NEGATIVE.
- Display the model's prediction score.
- Access the application through a Gradio web interface.

## Technology Stack

- **Python:** Application development
- **Hugging Face Transformers:** Model inference
- **DistilBERT:** Pretrained sentiment classification
- **PyTorch:** Neural-network execution
- **Gradio:** Web interface
- **Hugging Face Spaces:** Application hosting

## Model

`distilbert/distilbert-base-uncased-finetuned-sst-2-english`

This model is fine-tuned for binary sentiment classification on English text.

## Architecture

Customer Review → Gradio Interface → Python Function → Hugging Face Pipeline → Sentiment Prediction → Gradio Output

The pipeline handles tokenization, model inference, and postprocessing.

## Live Demo

[Launch the Customer Feedback Analyzer](https://huggingface.co/spaces/Amarnath1991/customer-feedback-analyzer)

*Deployment status should be verified before presenting this as a working demo.*

## Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python app.py
```

Open the local Gradio URL displayed in the terminal.

## Limitations

- Supports positive/negative sentiment classification rather than detailed emotion detection.
- Designed for English text.
- Mixed or sarcastic reviews may be misclassified.
- Prediction scores are not guarantees of correctness.
- Long inputs may be truncated.

## Planned Enhancements

- Bulk CSV review analysis
- Complaint categorization
- Sentiment distribution dashboard
- Downloadable analysis reports
- Model evaluation on representative customer feedback

## Business Value

The application demonstrates a starting point for automating customer feedback analysis. Future enhancements could help product and CX teams identify recurring issues and prioritize improvements.

## Author

AI Engineering Portfolio Project