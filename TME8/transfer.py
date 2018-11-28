import argparse
import os
import time

from PIL import Image

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

models.vgg.model_urls["vgg16"] = "http://webia.lip6.fr/~robert/cours/rdfia/vgg16-397923af.pth"
os.environ["TORCH_MODEL_ZOO"] = "/tmp/torch"
PRINT_INTERVAL = 50
CUDA = True

def get_dataset(batch_size, path):

    # Cette fonction permet de recopier 3 fois une image qui
    # ne serait que sur 1 channel (donc image niveau de gris)
    # pour la "transformer" en image RGB. Utilisez la avec
    # transform.Lambda
    def duplicateChannel(img):
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
        return img

    train_dataset = datasets.ImageFolder(path+'/train',
        transform=transforms.Compose([ 
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 456, 0.406), (0.229, 0.224, 0.225))
        ]))
    val_dataset = datasets.ImageFolder(path+'/test',
        transform=transforms.Compose([ 
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 456, 0.406), (0.229, 0.224, 0.225))
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=False, pin_memory=CUDA, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size, shuffle=False, pin_memory=CUDA, num_workers=2)

    return train_loader, val_loader


def extract_features(data, model):
    N = len(data) * data.batch_size
    W = 4096
    # TODO init features matrices
    X = torch.zeros((N, W))
    y = torch.zeros((N))
    for i, (input, target) in enumerate(data):
        if i % PRINT_INTERVAL == 0:
            print('Batch {0:03d}/{1:03d}'.format(i, len(data)))
        if CUDA:
            input = input.cuda()
        output = model(input)
        batch_size = input.shape[0]
        X[i*batch_size:(i+1)*batch_size] = output 
        y[i*batch_size:(i+1)*batch_size] = target
    return X, y

class VGG16relu7(nn.Module):
    def __init__(self, vgg16):
        super(VGG16relu7, self).__init__()
        # recopier toute la partie convolutionnelle
        self.features = nn.Sequential( *list(vgg16.features.children()))
        # garder une partie du classifieur, -2 pour s'arrêter à relu7
        self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def main(params):
    print('Instanciation de VGG16')
    vgg16 = models.vgg16(pretrained=True)

    print('Instanciation de VGG16relu7')
    model = VGG16relu7(vgg16)
    
    model.eval()
    if CUDA: # si on fait du GPU, passage en CUDA
        model = model.cuda()

    # On récupère les données
    print('Récupération des données')
    train, test = get_dataset(params.batch_size, params.path)
    #train, test = get_dataset(1, params.path)

    # Extraction des features
    print('Feature extraction')
    X_train, y_train = extract_features(train, model)
    X_test, y_test = extract_features(test, model)

    # TODO Apprentissage et évaluation des SVM à faire
    print('Apprentissage des SVM')
    accuracy = ...

if __name__ == '__main__':

    # Paramètres en ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='data/15SceneData', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='activate GPU acceleration')

    args = parser.parse_args()
    if args.cuda:
        CUDA = True
        cudnn.benchmark = True

    # parce que le mode éval continue de calculer les gradients,
    # ce qu'il ne faisait pas auparavant.
    with torch.no_grad():
        main(args)

    input("done")
