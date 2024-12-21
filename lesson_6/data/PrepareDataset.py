from datasets import load_dataset
from transformers import T5Tokenizer

dataset = load_dataset("squad")
#dataset = load_dataset("squad_v2")
#dataset = load_dataset("narrativeqa")
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
        padding="max_length",
        truncation=True,
        max_length=128
    )

    # Токенізуємо вихідний текст (targets)
    # Беремо першу відповідь зі списку 'answers' як ground truth
    answers = [ans["text"][0] if len(ans["text"]) > 0 else "" for ans in targets]

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            answers,
            padding="max_length",
            truncation=True,
            max_length=128
        )

    # Додаємо мітки до результатів
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# Токенізуємо датасет з використанням `map`
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Зберігаємо токенізований датасет на диск
tokenized_dataset.save_to_disk("../tokenized_dataset")

print("Розділені токенізовані дані збережено на диск")
print(tokenized_dataset["train"].column_names)