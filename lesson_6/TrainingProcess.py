import torch
import evaluate
from torch.optim import AdamW
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorWithPadding
from datetime import datetime
from lesson_6.BasicModelEvaluation import evaluate_model

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "xpu" if torch.xpu.is_available()
    else "cpu")
print(f"Використовується пристрій: {device}")

# Параметри
model_name = "t5-small"
learning_rate = 1e-5
batch_size = 16
epochs = 4

# Токенізатор
tokenizer = T5Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Модель
basic_model = T5ForConditionalGeneration.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Дані
squad_v2_dataset = load_from_disk("tokenized_squad_v2")
print("Датасет tokenized_squad_v2 завантажено з диска")

train_dataset = squad_v2_dataset["train"].select(range(10000))
validation_dataset = squad_v2_dataset["validation"]

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=data_collator)

# Метрики
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

print("Cтворено даталоадери")

# Навчання
num_batches = len(train_dataloader)
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(train_dataloader, start=1):
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] Обробка батчу {batch_idx} з {num_batches}")

        optimizer.zero_grad()
        input_ids = torch.stack([torch.LongTensor(ids) for ids in batch["input_ids"]]).to(device)
        attention_mask = torch.stack([torch.LongTensor(mask) for mask in batch["attention_mask"]]).to(device)
        labels = torch.stack([torch.LongTensor(label) for label in batch["labels"]]).to(device)

        # Передача даних у модель
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Генерація тексту для прикладу
        if batch_idx % 100 == 0:
            model.eval()
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids[:1],  # Перший приклад з батчу
                    attention_mask=attention_mask[:1],
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).replace("<pad>", "").strip()
                reference_text = tokenizer.decode(labels[0], skip_special_tokens=True)

                print(f"Generated: {generated_text}")
                print(f"Reference: {reference_text}")

                # Обчислення BLEU і ROUGE
                if len(generated_text) != 0 and len(reference_text) != 0:
                    bleu_score = bleu.compute(predictions=[generated_text], references=[[reference_text]])
                    rouge_score = rouge.compute(predictions=[generated_text], references=[reference_text])
                    print(f"BLEU: {bleu_score['bleu']}, ROUGE-L: {rouge_score['rougeL']}")
            model.train()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Епоха {epoch + 1} завершена, середні втрати: {avg_loss:.4f}")
    print(f"----------------------")

# Оцінка моделі
print("Оцінка моделі на валідаційному датасеті:")
basic_bleu_score, basic_rouge_score = evaluate_model(basic_model, validation_dataset, tokenizer, device)
print(f"Basic. BLEU: {basic_bleu_score['bleu']}, ROUGE-L: {basic_rouge_score['rougeL']}")
bleu_score, rouge_score = evaluate_model(model, validation_dataset, tokenizer, device)
print(f"Trained. BLEU: {bleu_score['bleu']}, ROUGE-L: {rouge_score['rougeL']}")
