from engine import tensor
from nn import MLP
from optim import Adam
from visualize import draw_dot

activations = [
    lambda x: x.relu(),
    lambda x: x.leaky_relu(alpha=0.01)
]
dropout_rates = [0.2, 0.1]

model = MLP([3, 4, 4, 1], activation_fns=activations, dropout_p=dropout_rates)
optimizer = Adam(model.parameters(), lr=0.05)

X = tensor([
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
])

Y = tensor([
    [1.0],
    [-1.0],
    [-1.0],
    [1.0]
])

epochs = 100

print("Gonna train w/ Adam...\n")

model.train()
for k in range(epochs):
    y_pred = model(X)
    loss = y_pred.mse_loss(Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if k % 10 == 0:
        print(f"Ep {k:<3} | loss: {loss.data[0]:.4f}")

model.eval()
print("\n--- all done — train finished ---")
print("tgt: [1.0, -1.0, -1.0, 1.0]")
print("out:", [round(val, 4) for val in y_pred.data])

print("\nmkng viz...")

graph = draw_dot(loss)
graph.render('macrograd_architecture', format='svg', view=True)