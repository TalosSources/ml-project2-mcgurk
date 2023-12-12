from mc_gurk_classification import *

N = 6
D = 2
C = 3

model = LogisticRegression(D, C)

X = torch.tensor(
    [
        [
            0,
            1,
        ],
        [
            -1,
            2.0,
        ],
        [
            15.0,
            1.0,
        ],
        [
            12.0,
            2.0,
        ],
        [1, 11],
        [2, 14],
    ]
)

Y = torch.nn.functional.one_hot(torch.tensor([0, 0, 1, 1, 2, 2])).float()
print(f"X = {X}")
print(f"Y = {Y}")

model.train_log_reg(X, Y, epochs=10000, learning_rate=0.01)
