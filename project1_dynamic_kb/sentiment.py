from transformers import pipeline

_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return _classifier

def detect_sentiment(text):
    classifier = get_classifier()
    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]

    if label == "POSITIVE":
        sentiment = "positive"
    elif label == "NEGATIVE":
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {"sentiment": sentiment, "confidence": round(score, 3)}

if __name__ == "__main__":
    tests = [
        "This is amazing, thank you so much!",
        "This is really frustrating and not working at all.",
        "The Eiffel Tower is in Paris."
    ]
    for t in tests:
        print(t, "->", detect_sentiment(t))