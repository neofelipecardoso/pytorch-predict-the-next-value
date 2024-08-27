import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim

key = "your api key"

api_url = f"https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol=AAPL&outputsize=compact&apikey={key}"


def pegar_data(api_url):
    response = requests.get(api_url)
    data = response.json()

    df = pd.DataFrame(data["data"])

    df = df[["date", "last"]]
    df["date"] = pd.to_datetime(df["date"])
    df["last"] = pd.to_numeric(df["last"])

    df = df.sort_values(by="date").reset_index(drop=True)
    return df


class OptionPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(OptionPredictor, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.hidden_layer(x)
        out = self.relu(out)
        out = self.output_layer(out)
        return out


def treinar_modelo(model, criterion, optimizer, inputs, targets, num_epochs=300):
    for epoch in range(num_epochs):
        model.train()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"vezes treinadas [{epoch+1}/{num_epochs}], Perdas: {loss.item():.4f}")


option_data = pegar_data(api_url)

input_size = 50
hidden_size = 100
output_size = 1

inputs = []
targets = []

option_data_values = option_data["last"].values

for i in range(len(option_data_values) - input_size):
    inputs.append(option_data_values[i : i + input_size])
    targets.append(option_data_values[i + input_size])

inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

model = OptionPredictor(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

treinar_modelo(model, criterion, optimizer, inputs, targets, num_epochs=300)

model.eval()
recent_data = torch.tensor(option_data_values[-input_size:], dtype=torch.float32).view(1, -1)
predicted_price = model(recent_data)
print(f"Preco previsto: {predicted_price.item():.4f}")
