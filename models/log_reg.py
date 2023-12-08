import torch
from tqdm import tqdm

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        print(f"LogisticRegression: input_dim={input_dim}, output_dim={output_dim}")
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        outputs = torch.softmax(self.linear(x), dim=-1)
        return outputs

    """
    @params :
        X : tensor of shape (N, d) where N is the amount of mcgurk videos (or subsamples of which), and d is perceiver latent_size (=512)
        Y : tensor of shape (N), of integers in [C], where C is the number of possible syllables.
        model : LogisticRegression model
    @return: nothing
    """

    def train_log_reg(self, X, Y, epochs=200000, learning_rate=0.01):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        progress_bar = tqdm(range(epochs), desc=f"Training...", leave=False, position=0)
        for iter in progress_bar:
            optimizer.zero_grad()
            outputs = self(X)
            # if(outputs.size(1) == 1):
            #    outputs = outputs.reshape(outputs.size(0))
            loss = criterion(outputs, Y)
            if iter % 5000 == 0:
                progress_bar.set_description(f"Training... (loss: {loss})", refresh=False)

            loss.backward()

            optimizer.step()
