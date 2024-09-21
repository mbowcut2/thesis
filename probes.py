import torch
import torch.nn.functional as F
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)
    
    def accuracy(self, y_hat, y):
        return (self.y_hat.argmax(dim=1) == y).float().mean()
    
    def predict(self, x):
        return self.linear(x).argmax(dim=1)
    
    def predict_proba(self, x):
        return F.softmax(self.linear(x), dim=1)
    
    def predict_logits(self, x):
        return self.linear(x)
    
    def get_weights(self):
        return self.linear.weight.data
    
    def get_bias(self):
        return self.linear.bias.data
    
    def set_weights(self, weights):
        self.linear.weight.data = weights

    def set_bias(self, bias):
        self.linear.bias.data = bias

    def get_grads(self):
        return self.linear.weight.grad.data, self.linear.bias.grad.data
