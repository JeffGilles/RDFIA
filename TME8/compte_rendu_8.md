1. On a pour les couches fc ≃ `7*7*512*4096 + 4096*4096 + 4096*1000` ≃ 124 millions de paramètres
2. La taille de sortie de la dernière couche est 1000 car il y a 1000 classes.
5. Parce que ça prendrait trop de temps
6. Des classes de ImageNet se retrouvent dans les scènes de 15Scenes, ainsi VGG ayant appris des paramètres permettant de reconnaitre ces classes il pourra distinguer plus facilement les scenes.
