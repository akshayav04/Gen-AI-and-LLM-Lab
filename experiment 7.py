from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)

# ---------------------------------
# 1. Domain-Specific Dataset
# ---------------------------------

data = {
    "text": [
        "The transformer model achieved excellent accuracy.",
        "Large Language Models are revolutionizing AI.",
        "The football team won the championship.",
        "The cricket match was exciting.",
        "Neural networks are widely used in deep learning.",
        "The player scored a brilliant goal.",
        "Machine learning improves decision making.",
        "The tennis tournament starts tomorrow."
    ],
    "label": [
        1, 1, 0, 0,
        1, 0, 1, 0
    ]
}

dataset = Dataset.from_dict(data)

# ---------------------------------
# 2. Load Tokenizer
# ---------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

# ---------------------------------
# 3. Tokenization
# ---------------------------------

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize)

dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label"
    ]
)

# ---------------------------------
# 4. Load Pretrained Model
# ---------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# ---------------------------------
# 5. Training Configuration
# ---------------------------------

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_steps=1,
    save_strategy="no",
    report_to="none"
)

# ---------------------------------
# 6. Fine-Tune Model
# ---------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

# ---------------------------------
# 7. Save Model
# ---------------------------------

trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# ---------------------------------
# 8. Load Fine-Tuned Model
# ---------------------------------

classifier = pipeline(
    "text-classification",
    model="./fine_tuned_model",
    tokenizer="./fine_tuned_model"
)

# ---------------------------------
# 9. Prediction
# ---------------------------------

text = "Generative AI models improve intelligent automation."

result = classifier(text)

labels = {
    "LABEL_0": "Sports",
    "LABEL_1": "Technology"
}

print("\nPrediction")
print("-" * 30)
print("Input:", text)
print("Predicted Class:", labels[result[0]["label"]])
print("Confidence Score:", round(result[0]["score"], 3))
