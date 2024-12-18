import evaluate
import torch
from datasets import load_from_disk
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Функція для оцінки моделі за BLEU та ROUGE
def evaluate_model(model, dataset, tokenizer, device):
    # Метрики
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    predictions = []
    references = []

    model.eval()
    with torch.no_grad():
        for example in dataset:

            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(device)
            labels = torch.tensor(example["labels"]).unsqueeze(0).to(device)

            # Генерація тексту
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace("<pad>", "").strip()
            reference_text = tokenizer.decode(labels[0], skip_special_tokens=True)

            predictions.append(generated_text)
            references.append([reference_text])

    # Обчислення BLEU і ROUGE
    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)

    return bleu_score, rouge_score