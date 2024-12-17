from datasets import load_dataset
from transformers import T5Tokenizer

squad_v2 = load_dataset("squad_v2")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
tokenizer.pad_token = tokenizer.eos_token


def preprocess_function(examples):
    # Формуємо вхідний текст для T5: завдання + питання + контекст
    inputs = [f"question: {q}  context: {c}" for q, c in zip(examples["question"], examples["context"])]
    # Вихідний текст — це правильна відповідь
    targets = examples["answers"]

    # Токенізуємо вхідний текст (inputs)
    model_inputs = tokenizer(
        inputs,
        padding="max_length",  # Паддинг до максимальної довжини
        truncation=True,
        max_length=128
    )

    # Токенізуємо вихідний текст (targets)
    # Беремо першу відповідь зі списку 'answers' як ground truth
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            [ans["text"][0] if len(ans["text"]) > 0 else "" for ans in targets],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    # Додаємо мітки до результатів
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["labels"] = [
        [label if label != tokenizer.pad_token_id else -100 for label in label_ids]
        for label_ids in model_inputs["labels"]
    ]

    return model_inputs


# Токенізуємо датасет з використанням `map`
tokenized_dataset = squad_v2.map(
    preprocess_function,
    batched=True,
    remove_columns=squad_v2["train"].column_names
)

# Зберігаємо токенізований датасет на диск
tokenized_dataset.save_to_disk("../tokenized_squad_v2")

print("Розділені токенізовані дані збережено на диск")