import numpy as np
import math
import torch

if torch.backends.mps.is_available():
    print("MPS is available.")
    device = torch.device("mps")
elif torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")
# print("device: {}".format(device))


class LinearClassifier:
    def __init__(self, learning_rate=0.05, epochs=500, reg=0.1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg = reg
        self.W = None
        self.b = None

    def train(self, X, Y):
        n_samples, n_features = X.shape
        n_classes = np.max(Y) + 1  # Assumes labels are 0-based

        weight_init = 2.0 / math.sqrt(n_features)

        self.W = weight_init * torch.randn(n_features, n_classes, device=device)
        self.b = torch.zeros(n_classes, device=device)

        self.W.requires_grad = True
        self.b.requires_grad = True

        optimizer = torch.optim.SGD(
            [self.W, self.b], lr=self.learning_rate, weight_decay=self.reg
        )

        criterion = torch.nn.CrossEntropyLoss()
        X = torch.from_numpy(X).float().to(device)
        Y = torch.from_numpy(Y).to(device)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            scores = torch.matmul(X, self.W) + self.b
            loss = criterion(scores, Y)
            # reg_loss =  self.reg * torch.sum(self.W**2)  # L2 regularization
            # loss += reg_loss

            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Loss {loss}")

        weight = self.W.cpu().detach().numpy()
        bias = self.b.cpu().detach().numpy()
        return weight, bias

    def predict(self, X):
        pass
