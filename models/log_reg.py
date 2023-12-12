import torch
from tqdm import tqdm


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        # print(f"LogisticRegression: input_dim={input_dim}, output_dim={output_dim}") # DEBUG
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)
        with torch.no_grad():
            self.linear.bias.zero_()
            self.linear.weight.zero_()

    def forward(self, x):
        return self.linear(x)

    """
    @params :
        X : tensor of shape (N, d) where N is the amount of mcgurk videos (or subsamples of which), and d is perceiver latent_size (=512)
        Y : tensor of shape (N), of integers in [C], where C is the number of possible syllables.
        model : LogisticRegression model
    @return: nothing
    """

    def train_log_reg(self, X, Y, epochs=200000, learning_rate=0.01):
        # Define loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        #optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, nesterov=True, momentum=0.9)

        progress_bar = tqdm(range(epochs), desc=f"Training...", leave=False, position=0)
        for iter in progress_bar:
            # Forward pass
            optimizer.zero_grad()
            outputs = self(X)

            # Compute loss
            loss = criterion(outputs, Y)

            if iter % 5000 == 0:
                # Print progress
                progress_bar.set_description(
                    f"Training... (loss: {loss})", refresh=False
                )

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()


    def predict(self, X):
        return(torch.nn.functional.softmax(self.linear(X))) 
