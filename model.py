import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv')

x_raw = data[['old_price', 'area', 'quality']].values
x_mean = x_raw.mean(axis=0)
x_std = x_raw.std(axis=0)

x = (x_raw - x_mean) / x_std 

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(data['new_price'].values, dtype=torch.float32).view(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = nn.Linear(3, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 5000

losses = []

for epoch in range(epochs):
    predictions = model(x_train)
    loss = criterion(predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.2f}')

y_pred = model(x_test)
test_loss = criterion(y_pred, y_test)
print(f'Test Loss: {test_loss.item():.4f}')

def predict_price(current_price: float, area: float, quality: float) -> float:
    model.eval()
    input_data = torch.tensor([[current_price, area, quality]], dtype=torch.float32)
    x_mean_tensor = torch.tensor(x_mean, dtype=torch.float32)
    x_std_tensor = torch.tensor(x_std, dtype=torch.float32)
    input_data = (input_data - x_mean_tensor) / x_std_tensor 
    with torch.no_grad():
        prediction = model(input_data)
    return round(prediction.item(), 3)