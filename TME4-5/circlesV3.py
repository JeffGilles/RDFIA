import math
import torch
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt
from tme5 import CirclesData

def init_model(nx, nh, ny):
    
    model = torch.nn.Sequential(
        torch.nn.Linear(nx, nh),
        torch.nn.Tanh(),
        torch.nn.Linear(nh, ny),
        torch.nn.Sigmoid()
    )
    loss = torch.nn.MSELoss()
    
    return model, loss


def loss_accuracy(Yloss, Yhat, Y):
    
    _, indsYh = torch.max(Yhat, 1)
    _, indsY = torch.max(Y, 1)
    acc = (indsYh == indsY).sum().float()/len(Y)
    #L = (- Y * torch.log(Yhat) - (1 - Y) * torch.log(1 - Yhat)).sum()
    #L = (Y * torch.log(Yhat)).sum(dim=1).mean()
    L = loss(Yloss, Y)
    
    return L, acc


def sgd(model, eta):
    with torch.no_grad():
        for param in model.parameters():
            param -= eta * param.grad
        model.zero_grad()


if __name__ == '__main__':

    # init
    data = CirclesData()
    data.plot_data()
    N = data.Xtrain.shape[0]
    Nbatch = 10
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    eta = 0.002

    # Premiers tests, code à modifier
    model, loss = init_model(nx, nh, ny)
    Yloss = model(data.Xtrain)
    #Yhat =
    L, _ = loss_accuracy(Yloss, Yhat, data.Ytrain)
    L.backward()
    sgd(model, eta)

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
            #Yhat = 
            Yloss = model[data.Xtrain]
            L, _ = loss_accuracy(Yloss, Yhat, Ybatch)
            L.backward()
            sgd(model, eta)
        #YlossTrain = 
        #YlossTest = 
        #Yhat_train =
        #Yhat_test =
        L, acc = loss_accuracy(Yloss, Yhat, data.Ytrain)
        L_test, acc_test = loss_accuracy(Yhat_test, data.Ytest)
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
