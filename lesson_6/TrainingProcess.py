import torch
from torch.optim import AdamW
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, DataCollatorWithPadding

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "xpu" if torch.xpu.is_available()
    else "cpu")

model = GPT2LMHeadModel.from_pretrained("gpt2")
optimizer = AdamW(model.parameters(), lr=5e-5)

# Завантаження токенізованих даних генерації тексту
squad_v2_dataset = load_from_disk("tokenized_squad_gpt2")
print("Датасет tokenized_squad_gpt2 завантажено з диска")

train_dataset = squad_v2_dataset["train"]
validation_dataset = squad_v2_dataset["validation"]


train_dataloader = DataLoader(squad_v2_dataset["train"], batch_size=8, shuffle=True)
validation_dataloader = DataLoader(squad_v2_dataset["validation"], batch_size=8)

print("Cтворено даталоадери")

model.train()
for epoch in range(1):
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = torch.stack([torch.LongTensor(ids) for ids in batch["input_ids"]]).to(device)
        attention_mask = torch.stack([torch.LongTensor(mask) for mask in batch["attention_mask"]]).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Епоха завершена: Втрата {loss.item()}")



