from mc_gurk_classification import *

N = 5
D = 2
C = 2

model = LogisticRegression(D, C)

X = torch.tensor([
    [1., 2.,],
    [0., 2.,],
    [1., 0.,],
    [12., 7.,],
    [8., 15],
])

Y = torch.nn.functional.one_hot(torch.tensor([0, 0, 0, 1, 1])).float()
print(f"X = {X}")
print(f"Y = {Y}")

model.train_log_reg(X, Y, epochs=1000, learning_rate=0.01)

