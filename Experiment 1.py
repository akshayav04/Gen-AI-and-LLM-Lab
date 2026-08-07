from transformers import GPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "Artificial Intelligence is"

inputs = tokenizer.encode(prompt, return_tensors="pt")# Generate text
outputs = model.generate(
 inputs,
 max_length=100,
 num_return_sequences=1,
 temperature=0.7,
 do_sample=True
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
