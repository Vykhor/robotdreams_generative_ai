import torch.nn as nn

class MnistModel(nn.Module):
  def __init__(self, input_size, output_classes_number, hidden_features, dropout):
    super(MnistModel, self).__init__()
    self.input = nn.Linear(input_size, hidden_features)
    self.relu = nn.ReLU()
    self.hidden1 = nn.Linear(hidden_features, hidden_features)
    self.hidden2 = nn.Linear(hidden_features, hidden_features)
    self.hidden3 = nn.Linear(hidden_features, hidden_features)
    self.output = nn.Linear(hidden_features, output_classes_number)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.input(x)
    x = self.relu(x)
    x = self.hidden1(x)
    x = self.dropout(x)
    x = self.relu(x)
    x = self.hidden2(x)
    x = self.dropout(x)
    x = self.relu(x)
    x = self.hidden3(x)
    x = self.output(x)
    return x

def create_model(input_size, output_classes_number, hidden_features, dropout):
    model = MnistModel(input_size=input_size, output_classes_number=output_classes_number, hidden_features=hidden_features, dropout=dropout)
    print(f"Created model: {model}")
    return model