#!/usr/bin/env python
# coding: utf-8

# In[25]:


get_ipython().system('pip install math')
get_ipython().system('pip install pandas_datareader')
get_ipython().system('pip install numpy')
get_ipython().system('pip install yfinance')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install tkinter')


# In[30]:


# Importation des bibliothèques nécessaires
import math
from pandas_datareader import data as pdr
import numpy as np
import yfinance as yf
yf.pdr_override()
from datetime import date
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
plt.style.use('fivethirtyeight')


# In[31]:


# Calcul du rendement total et la volatilité annuel via le df et prix de cloture
def calculate_returns(df):
    # calcul le retour quotidien
    df['daily_return'] = df['Close'].pct_change()
    # calcul le rendement total sur investissement
    retour_inves = np.prod(1 + df['daily_return']) - 1
    # calcul de la volatilité annualisé en sachant que 252 = le nombre de jour moyen de trading par an
    volatilite_an = df['daily_return'].std()*np.sqrt(252)
    return retour_inves, volatilite_an

# Pour refresh les visualisations graphiques s'il y a plusieurs input
def clear_canvas():
    for widget in window.winfo_children():
        if isinstance(widget, Canvas):
            widget.destroy()

# Plot sur canvas (via matploblib)
def plot_on_canvas(figure):
    canvas = FigureCanvasTkAgg(figure, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()


# In[32]:


# Prediction
def plot_prediction(symbol, date_debut, date_fin):
    
    try:
        # Récupérer les données de l'actif financier
        df = pdr.get_data_yahoo(symbol, start=date_debut, end=date_fin)
        retour_inves, volatilite_an = calculate_returns(df)
        # + Rendement total et Volatilité annualisée
        summary_text.set(f'Rendement total: {retour_inves*100:.2f}%\n'
                         f'Volatilité annualisée: {volatilite_an*100:.2f}%')
    except:
        # Message d'erreur affiché si l'actif financier n'existe pas (ex : GOOGLE au lieu de GOOGL)
        error_text.set("L'actif financier n'existe pas ou n'est pas dans la base de données.")
        return None

    # Création d'une nouvelle dataframe avec le prix de cloture seulement utilisant .filter
    data = df.filter(['Close'])
    
    # Conversion de la dataframe en tableau pour numpy
    dataset = data.values
    
    # Calcul du nombre de lignes pour entrainer le modele LSTM
    tendance_dlong = math.ceil(len(dataset) * 0.8)
    
    # Echelonnage de la donnée sur [0,1]
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Creation de la tendance de la data echelonnée
    tendance_data = scaled_data[0:tendance_dlong , :]
    tendance_x = []
    tendance_y = []

    for i in range(60, len(tendance_data)):
        tendance_x.append(tendance_data[i-60:i, 0])
        tendance_y.append(tendance_data[i, 0])

    # Conversion de x_train et y_train en tableau numpy pour LSTM
    tendance_x = np.array(tendance_x)
    tendance_y = np.array(tendance_y)

    # Reshape de x_train pour le modele LSTM (3 dimensions)
    tendance_x = np.reshape(tendance_x, (tendance_x.shape[0], tendance_x.shape[1], 1))

    # Création du modèle LSTM
    lstm = Sequential()
    lstm.add(LSTM(50, return_sequences=True, input_shape= (tendance_x.shape[1], 1)))
    lstm.add(LSTM(50, return_sequences=False))
    lstm.add(Dense(25))
    lstm.add(Dense(1))

    lstm.compile(optimizer='adam', loss='mean_squared_error')

    lstm.fit(tendance_x, tendance_y, batch_size=1, epochs=1)

    test_data = scaled_data[tendance_dlong - 60:, :]
    test_x = []
    test_y = dataset[tendance_dlong:, :]

    for i in range(60, len(test_data)):
        test_x.append(test_data[i-60:i, 0])

    # 3D shape
    test_x = np.array(test_x)
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    #Prediction
    predictions = lstm.predict(test_x)
    predictions = scaler.inverse_transform(predictions)

    tendance = data[:tendance_dlong]
    pred = data[tendance_dlong:]
    pred['Predictions'] = predictions

    # Plot des données 
    figure = plt.Figure(figsize=(15,8), dpi=100)
    ax = figure.add_subplot(111)
    ax.plot(tendance['Close'])
    ax.plot(pred[['Close', 'Predictions']])
    ax.set_title('LSTM Prevision')
    ax.set_xlabel('Date')
    ax.set_ylabel('Prix de Cloture ($)')
    ax.legend(['Tendance', 'Valeur', 'Prevision'], loc='upper left')

    return figure


# In[33]:


# Configuration de l'interface utilisateur Tkinter
def predict():
    clear_canvas()
    error_text.set("")
    figure = plot_prediction(entry_text.get(), date_debut_text.get(), date_fin_text.get())
    
    if figure:
        plot_on_canvas(figure)


# In[34]:


# Initialisation de la fenêtre Tkinter
window = Tk()
window.title('Prédiction du prix de clôture des actifs financiers')

#Création visuelle

entry_text = StringVar()
entry_label = Label(window, text="Actif Financier (ex : AAPL pour Apple)")
entry_label.pack()
entry = Entry(window, textvariable=entry_text)
entry.pack()

date_debut_text = StringVar()
date_debut_label = Label(window, text="Date de départ (YYYY-MM-DD):")
date_debut_label.pack()
date_debut_entry = Entry(window, textvariable=date_debut_text)
date_debut_entry.pack()

date_fin_text = StringVar()
date_fin_label = Label(window, text="Date de fin (YYYY-MM-DD):")
date_fin_label.pack()
date_fin_entry = Entry(window, textvariable=date_fin_text)
date_fin_entry.pack()

button = Button(window, text='Prédire', command=predict)
button.pack()

summary_text = StringVar()
summary_label = Label(window, textvariable=summary_text)
summary_label.pack()

error_text = StringVar()
error_label = Label(window, textvariable=error_text, fg="red")
error_label.pack()

window.mainloop()


# In[ ]:




