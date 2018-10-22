from tme5 import CirclesData
import math
import torch
from torch.distributions.normal import Normal
from torch import autograd
import matplotlib.pyplot as plt

#avec torch.autograd

def init_params(nx, nh, ny):
    params = {}

    params["Wh"] = torch.zeros([nh, nx]).normal_(0, 0.3)
    params["Wy"] = torch.zeros([ny, nh]).normal_(0, 0.3)
    params["bh"] = torch.zeros([nh]).normal_(0, 0.3)
    params["by"] = torch.zeros([ny]).normal_(0, 0.3)
    params["Wh"].requires_grad = True
    params["Wy"].requires_grad = True
    params["bh"].requires_grad = True
    params["by"].requires_grad = True
    
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
    L = - torch.sum(Y * torch.log(Yhat))
    #L = (- Y * torch.log(Yhat) - (1 - Y) * torch.log(1 - Yhat)).sum()
    #L = (Y * torch.log(Yhat)).sum(dim=1).mean()
    
    return L, acc


def sgd(params, eta):    
    with torch.no_grad():
        for param in params.keys():
            params[param] -= eta * params[param].grad
            params[param].grad.zero_()
    
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
    L.backward() #calcule les gradients
    params = sgd(params, eta)
    
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
            L.backward()
            params = sgd(params, eta)
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
