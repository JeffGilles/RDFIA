Nous avons pu implémenter toutes les méthodes de l'énoncé (sans bonus).

La normalisation consiste à centrer toutes les données autour de 0, cela permet d'apprendre plus rapidement, car si une feature a des données très grandes et que les autres features ont des données plus petites, les paramètres associés aux autres features apprendront plus lentement.

L'augmentation artificielle des exemples d'apprentissages permet de disposer d'un ensemble de données plus grand pour l'apprentissage, avec plus de variance qu'un simple tirage avec remise des exemples.

L'optimisation permet d'adapter efficacement la valeur d'epsilon et des gradients au cours de l'apprentissage. Un epsilon fixe se révèle souvent soit trop faible au début soit trop fort à la fin. On peut pour régler ça ajouter de l'inertie au gradient pour réduire la variance induite par une descente stochastique ou réduire progressivement le pas d'apprentissage.

Le dropout permet d'éviter le sur-apprentissage en désactivant pendant la phase d'entraînement un certain nombre de neurones, ce qui permet de réduire le nombre de paramètres, et par conséquent l'expressivité du modèle, (de plus chaque epoch devrait être aussi plus rapide).

Même si on normalise les exemples en entrée, il n'est pas dit qu'à la sortie de la première couche (et des couches suivantes) les données obtenues soient elles aussi normalisée. La batch-normalisation permet de pallier ce problème en s'assurant que les données d'entrée de chaque couche soient normalisées. 

