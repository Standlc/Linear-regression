import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("./data.csv")

m = len(data.values)
mileage = np.array([set[0] for set in data.values])
price = np.array([set[1] for set in data.values])

# NORMALISE THE DATA
max_mileage = mileage.max()
mileage = mileage / max_mileage
price = price / max_mileage

learning_rate = 0.1
epochs = 10000
teta0 = 0
teta1 = 0

losses = []

def train(teta0, teta1):
    for _ in range(epochs):
        loss = sum(((set[0] * teta1 + teta0) - set[1]) ** 2 for set in zip(mileage, price))
        losses.append(loss)
        
        predictions = mileage * teta1 + teta0

        grad_teta0 = (1 / m) * np.sum(predictions - price)
        grad_teta1 = (1 / m) * np.sum((predictions - price) * mileage)

        teta0 -= learning_rate * grad_teta0
        teta1 -= learning_rate * grad_teta1
        
    return teta0, teta1

teta0, teta1 = train(teta0, teta1)

# DENORMALISE THE DATA
teta0 *= max_mileage
mileage = mileage * max_mileage
price = price * max_mileage


model_graph = plt.subplot2grid((1, 2), (0, 0))
loss_graph = plt.subplot2grid((1, 2), (0, 1))

model_graph.set_title("Model")
x = np.linspace(mileage.min(), mileage.max(), 2)
model_graph.plot(mileage, price, '.')
model_graph.plot(x, teta1 * x + teta0)

loss_graph.set_title("Loss")
loss_graph.plot(range(0, len(losses)), losses)

plt.show()