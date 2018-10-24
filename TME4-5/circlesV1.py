import math
import torch
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt
from tme5 import CirclesData

def init_params(nx, nh, ny):
    params = {}

    params["Wh"] = torch.zeros([nh, nx]).normal_(0, 0.3)
    params["Wy"] = torch.zeros([ny, nh]).normal_(0, 0.3)
    params["bh"] = torch.zeros([nh]).normal_(0, 0.3)
    params["by"] = torch.zeros([ny]).normal_(0, 0.3)

    return params


def forward(params, X):
    outputs = {}

    outputs["X" ] = X
    outputs["htilde"] = X.matmul(params["Wh"].t()) + params["bh"]#.resize(params["bh"].shape[0], 1)
    outputs["h"] = torch.tanh(outputs["htilde"])
    outputs["ytilde"] = outputs["h"].matmul(params["Wy"].t()) + params["by"]
    # ? normaliser x pb explosion softmax
    #x = x / x.max()
    #ytilde = outputs["ytilde"] / outputs["ytilde"].max()
    outputs["yhat"] = torch.exp(outputs["ytilde"]) / torch.exp(outputs["ytilde"]).sum(dim=1).resize(X.shape[0], 1)

    return outputs['yhat'], outputs

def loss_accuracy(Yhat, Y):
    _, indsYh = torch.max(Yhat, 1)
    _, indsY = torch.max(Y, 1)
    acc = (indsYh == indsY).sum().float()/len(Y)
    #L = (- Y * torch.log(Yhat) - (1 - Y) * torch.log(1 - Yhat)).sum()
    #L = (Y * torch.log(Yhat)).sum(dim=1).mean()
    L = - torch.sum(Y * torch.log(Yhat))
    return L, acc

def backward(params, outputs, Y):
    grads = {}

    delta_yh = outputs["yhat"] - Y
    delta_hh = delta_yh.mm(params["Wy"]) * (1 - outputs["h"] **2)
    grads["Wy"] = delta_yh.t().mm(outputs["h"])
    grads["Wh"] = delta_hh.t().mm(outputs["X"])
    grads["by"] = delta_yh.sum(dim=0)
    #grads["by"] = grads["by"].resize(grads["by"].shape[0], 1)
    grads["bh"] = delta_hh.sum(dim=0)
    #grads["bh"] = grads["bh"].resize(grads["bh"].shape[0], 1)

    return grads

def sgd(params, grads, eta):
    for param in params.keys():
        params[param] -= grads[param]*eta
    return params



if __name__ == '__main__':

    # init
    data = CirclesData()
    data.plot_data()
    N = data.Xtrain.shape[0]
    Nbatch = 10
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    eta = 0.01

    # Premiers tests, code à modifier
    params = init_params(nx, nh, ny)
    Yhat, outs = forward(params, data.Xtrain)
    L, _ = loss_accuracy(Yhat, data.Ytrain)
    grads = backward(params, outputs=outs, Y=data.Ytrain)
    params = sgd(params, grads, eta)

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
            Yhat, outs = forward(params, Xbatch)
            L, _ = loss_accuracy(Yhat, Ybatch)
            grads = backward(params, outputs=outs, Y=Ybatch)
            params = sgd(params, grads, eta)
        Yhat, _ = forward(params, data.Xtrain)
        Yhat_test, _ = forward(params, data.Xtest)
        L, acc = loss_accuracy(Yhat, data.Ytrain)
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
