#A. Sentiment Analysis
from transformers import pipeline
sentiment_analyzer = pipeline("sentiment-analysis")
text = "The Generative AI workshop was extremely informative and useful."
result = sentiment_analyzer(text)

print(result)


#B. Document Classification
from transformers import pipeline
classifier = pipeline("zero-shot-classification")
document = """
Artificial Intelligence and Machine Learning are transforming
industries through automation and intelligent decision-making.
"""

labels = ["Technology", "Sports", "Politics", "Entertainment"]

result = classifier(document, labels)

print(result)
