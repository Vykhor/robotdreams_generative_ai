from datasets import load_dataset
from transformers import GPT2Tokenizer

squad_v2 = load_dataset("squad_v2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    # Об'єднуємо питання та контекст в один текст
    inputs = [f"Question: {q}\nContext: {c}\nAnswer:" for q, c in zip(examples["question"], examples["context"])]
    # Токенізуємо текст
    tokenized_inputs = tokenizer(
        inputs,
        padding="max_length",     # Додає паддинг до максимальної довжини
        truncation=True,          # Обрізає текст до максимальної довжини
        max_length=512,           # GPT-2 має обмеження на довжину
        return_tensors="pt"       # Повертає тензори PyTorch
    )
    return tokenized_inputs

tokenized_dataset = squad_v2.map(preprocess_function, batched=True, remove_columns=squad_v2["train"].column_names)

tokenized_dataset.save_to_disk("../tokenized_squad_gpt2")

print("Розділені токенізовані дані збережено на диск")