import evaluate

def generate_text(model, tokenizer, input_ids, attention_mask):
    generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

def evaluate_model(model, tokenizer, dataset, device):
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    predictions = []
    references = []

    for example in dataset:
        question = example["question"]
        context = example["context"]
        input_text = f"Question: {question}\nContext: {context}\nAnswer:"

        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        generated_text = generate_text(model, tokenizer, input_ids, attention_mask)
        reference_text = example["answers"]["text"][0]  # Візьмемо першу відповідь

        predictions.append(generated_text)
        references.append([reference_text])

    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)

    return bleu_score, rouge_score