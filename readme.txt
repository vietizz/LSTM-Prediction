L'application est écrite en Python et utilise les bibliothèques suivantes :

    1. pandas-datareader pour récupérer les données financières
    2. numpy pour le calcul numérique
    3. tensorflow pour le machine learning
    4. matplotlib pour la visualisation des données
    5. Tkinter pour l'interface utilisateur

Comment l'utiliser

    1. Ouvrez le fichier .py. Executer toutes les cellules lorsque vous arrivez à l'exécution de la dernière vous verrez une interface utilisateur avec plusieurs champs de saisie.
    2. Dans le champ "Actif Financier", entrez le symbole de l'actif financier que vous souhaitez analyser. Par exemple, pour Apple, le symbole est AAPL.
    3. Dans les champs "Date de départ" et "Date de fin", entrez la plage de dates sur laquelle vous souhaitez baser votre prédiction.
    4. Cliquez sur le bouton "Prédire". L'application récupérera les données de l'actif financier, entraînera le modèle LSTM et affichera le graphique de prédiction.
    5. Si vous voulez faire une autre prédiction, cliquez simplement sur le bouton "Prédire" à nouveau. L'ancien graphique sera effacé et un nouveau graphique sera affiché.

Fonctionnement interne

L'application commence par récupérer les données de l'actif financier à l'aide de la bibliothèque pandas-datareader. Ensuite, elle calcule le rendement quotidien et la volatilité annuelle de l'actif.