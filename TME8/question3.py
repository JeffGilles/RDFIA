import pickle, PIL
import numpy as np
from PIL import Image

imagenet_classes = pickle.load(open('data/imagenet_classes.pkl', 'rb')) # chargement des classes

# on normalise les données pour initialiser les paramètres de la première couche et avoir de bonnes conditions de convergence.
#img = PIL.Image.open("cat.jpg")
img = PIL.Image.open('data/15SceneData/train/forest/image_0028.jpg')
img = img.resize((224, 224), PIL.Image.BILINEAR)
img = np.array(img, dtype=np.float32)
img = img/255
img = img.transpose((2, 0, 1))
# TODO preprocess image
#img.transform
img.Resize((224, 224))
'''
transform=transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 456, 0.406), (0.229, 0.224, 0.225))
        transforms.ToTensor()
    ]))
''' 
img = np.expand_dims(img, 0) # transformer en batch contenant une image
x = torch.Tensor(img)
y = ... # TODO calcul forward
y = y.numpy() # transformation en array numpy
# TODO récupérer la classe prédite et son score de confiance
