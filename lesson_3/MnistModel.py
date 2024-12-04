import torch.nn as nn

class MnistModel(nn.Module):
  def __init__(self, input_size, output_classes_number, hidden_features, dropout):
    super(MnistModel, self).__init__()
    self.input = nn.Linear(input_size, hidden_features)
    self.bn_input = nn.BatchNorm1d(hidden_features)
    self.relu = nn.ReLU()

    self.hidden1 = nn.Linear(hidden_features, hidden_features)
    self.bn_hidden1 = nn.BatchNorm1d(hidden_features)
    self.hidden2 = nn.Linear(hidden_features, hidden_features)
    self.bn_hidden2 = nn.BatchNorm1d(hidden_features)
    self.hidden3 = nn.Linear(hidden_features, hidden_features)
    self.bn_hidden3 = nn.BatchNorm1d(hidden_features)

    self.dropout = nn.Dropout(dropout)

    self.output = nn.Linear(hidden_features, output_classes_number)

  def forward(self, x):
    x = x.view(x.size(0), -1)

    # вхідний шар
    x = self.input(x)
    x = self.bn_input(x)
    x = self.relu(x)

    # прихований шар 1
    x = self.hidden1(x)
    x = self.bn_hidden1(x)
    x = self.dropout(x)
    x = self.relu(x)

    # прихований шар 2
    x = self.hidden2(x)
    x = self.bn_hidden2(x)
    x = self.dropout(x)
    x = self.relu(x)

    # прихований шар 3
    x = self.hidden3(x)
    x = self.bn_hidden3(x)
    x = self.dropout(x)
    x = self.relu(x)

    # вихідний шар
    x = self.output(x)
    return x

def create_model(input_size, output_classes_number, hidden_features, dropout):
    model = MnistModel(input_size=input_size, output_classes_number=output_classes_number, hidden_features=hidden_features, dropout=dropout)
    print(f"Created model: {model}")
    return model