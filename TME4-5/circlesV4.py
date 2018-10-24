import math
import torch
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt
from tme5 import CirclesData

def init_model(nx, nh, ny, eta):
    
    model = torch.nn.Sequential(
        torch.nn.Linear(nx, nh),
        torch.nn.Tanh(),
        torch.nn.Linear(nh, ny),
        torch.nn.Sigmoid()
    )
    loss = torch.nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=eta)
    
    return model, loss, optim


def loss_accuracy(Yhat, Y, loss):
    
    _, indsYh = torch.max(Yhat, 1)
    _, indsY = torch.max(Y, 1)
    acc = (indsYh == indsY).sum().float()/len(Y)
    #L = (- Y * torch.log(Yhat) - (1 - Y) * torch.log(1 - Yhat)).sum()
    #L = (Y * torch.log(Yhat)).sum(dim=1).mean()
    L = loss(Yhat, Y)
    
    return L, acc


if __name__ == '__main__':

    # init
    data = CirclesData()
    data.plot_data()
    N = data.Xtrain.shape[0]
    Nbatch = 10
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    eta = 0.1

    # Premiers tests, code à modifier
    model, loss, optim = init_model(nx, nh, ny, eta)
    Yhat = model(data.Xtrain)
    L, _ = loss_accuracy(Yhat, data.Ytrain, loss)
    optim.zero_grad()
    L.backward()
    optim.step()

    # TODO apprentissage
    epochs = 500
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []
    for epoch in range(epochs):
        print(epoch)
        for j in range(int(N/Nbatch)):
            Xbatch = data.Xtrain[Nbatch*j:Nbatch*(j+1), :]
            Ybatch = data.Ytrain[Nbatch*j:Nbatch*(j+1), :]
            Yhat = model(Xbatch)
            L, _ = loss_accuracy(Yhat, Ybatch, loss)
            optim.zero_grad()
            L.backward()
            optim.step()
        Yhat_train = model(data.Xtrain)
        Yhat_test = model(data.Xtest)
        L, acc = loss_accuracy(Yhat_train, data.Ytrain, loss)
        L_test, acc_test = loss_accuracy(Yhat_test, data.Ytest, loss)
        train_accuracies.append(acc)
        train_losses.append(L)
        test_accuracies.append(acc_test)
        test_losses.append(L_test)
    
    plt.figure()
    plt.plot(range(epochs), train_accuracies, test_accuracies)
    plt.legend(["train", "test"])
    plt.figure()
    plt.plot(range(epochs), train_losses, test_losses)
    plt.legend(["train", "test"])
    # attendre un appui sur une touche pour garder les figures
    input("done")
