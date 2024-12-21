import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorWithPadding, get_scheduler
from datetime import datetime

from lesson_6.ModelEvaluation import evaluate_model, generate_example

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu")
print(f"Використовується пристрій: {device}")

# Параметри
model_name = "t5-small"
learning_rate = 1e-6
weight_decay = 0.01
batch_size = 16
epochs = 5

# Токенізатор
tokenizer = T5Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Модель
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Дані
dataset = load_from_disk("tokenized_dataset")
print("Датасет tokenized_squad_v2 завантажено з диска")

train_dataset = dataset["train"].select(range(int(len(dataset["train"]) * 0.3)))
print(f"train data: {train_dataset}")

validation_dataset = dataset["validation"]
print(f"validation data: {validation_dataset}")

# Оцінка перед навчанням
#basic_bleu_score, basic_rouge_score = evaluate_model(model, validation_dataset, tokenizer, device)
#print("Оцінка моделі на валідаційному датасеті:")
#print(f"Basic. BLEU: {basic_bleu_score['bleu']}, ROUGE-L: {basic_rouge_score['rougeL']}")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=data_collator)

num_training_steps = len(train_dataloader) * epochs
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

print("Cтворено даталоадери")

# Парааметри для ранньої зупинки
best_loss = float('inf')
patience = 2
counter = 0

# Навчання
num_batches = len(train_dataloader)
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(train_dataloader, start=1):
        current_time = datetime.now().strftime("%H:%M:%S")
        #print(f"[{current_time}] Обробка батчу {batch_idx} з {num_batches}")

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Передача даних у модель
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"")
    print(f"Епоха {epoch + 1} завершена, середні втрати: {avg_loss:.4f}")
    print(f"")

    generated_text, reference_text = generate_example(model=model,
                                                      tokenizer=tokenizer,
                                                      input_ids=input_ids,
                                                      attention_mask=attention_mask,
                                                      labels=labels)
    print(f"Generated: {generated_text}")
    print(f"Reference: {reference_text}")
    print(f"")
    avg_loss = total_loss / len(train_dataloader)

    # Перевірка ранньої зупинки
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in validation_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

    val_loss /= len(validation_dataloader)
    print(f"Валідаційні втрати: {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        print("Нова найкраща модель!")
    else:
        counter += 1
        print(f"Покращення не відбулося ({counter}/{patience})")
        if counter >= patience:
            print("Рання зупинка.")
            break

    print(f"--------------------------------------------")

# Оцінка після навчанням
bleu_score, rouge_score = evaluate_model(model, validation_dataset, tokenizer, device)

print("Оцінка моделі на валідаційному датасеті:")
print(f"Trained. BLEU: {bleu_score['bleu']}, ROUGE-L: {rouge_score['rougeL']}")
