from transformers import pipeline, set_seed
from datasets import load_from_disk
from torch.utils.data import DataLoader

# Завантаження пайплайну генерації тексту
generator = pipeline('text-generation', model='gpt2')

# Завантаження токенізованих даних генерації тексту
squad_v2_dataset = load_from_disk("tokenized_squad_gpt2")
print("Датасет tokenized_squad_gpt2 завантажено з диска")

train_dataset = squad_v2_dataset["train"]
validation_dataset = squad_v2_dataset["validation"]

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=8)

print("Cтворено даталоадери")





