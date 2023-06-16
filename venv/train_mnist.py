# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:40:49 2018

@author: user
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_mean
import network
from keras.datasets import mnist
from PIL import Image

#Dodawanie szumu do obrazka
def get_corrupted_input(input, corruption_level):
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted

#Zmiana kształtu tablicy
def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

#Generowanie siatki w plot i ustawianie odpowienich obrazków do wyświetlenia
def plot(data, test, predicted, figsize=(3, 3)):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]
    
    fig, axarr = plt.subplots(len(test), 2, figsize=figsize)
    for i in range(len(test)):
        if i==0:
            axarr[i, 0].set_title("Test data")
            axarr[i, 1].set_title('Output data')

        axarr[i, 0].imshow(test[i])
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(predicted[i])
        axarr[i, 1].axis('off')
            
    plt.tight_layout()
    plt.savefig("result_mnist.png")
    plt.show()

def preprocessing(img):
    w, h = img.shape
    # Thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1 # Boolian to int
    
    # Reshape
    flatten = np.reshape(shift, (w*h))
    return flatten

def main():
    #Pobieranie obrazka z pliku i przerobienie go na obraz w odcieniach szarości
    #I przerobienie na tablicę numpy
    image = Image.open("5.png")
    image_gray = image.convert('L')
    image_gray = image_gray.resize((28, 28))
    image_aray = np.array(image_gray) / 255.0


    # Wczytywanie obrazków z w tablicach numpy do listy
    data = []
    data.append(image_aray)

    #Preprocessing
    print("Start to data preprocessing...")
    data = [preprocessing(d) for d in data]
    
    # Tworzenie sieci Hopfielda
    model = network.HopfieldNetwork()
    model.train_weights(data)
    
    # Tworzenie listy obrazów do testów
    test = []
    test.append(image_aray)

    #Dodanie szumu do obrazków i dodanie do listy testów.
    test.append(get_corrupted_input(image_aray, 0.6))
    test = [preprocessing(d) for d in test]

    #Wyświetlanie wyników
    predicted = model.predict(test, threshold=50, asyn=True)
    print("Show prediction results...")
    plot(data, test, predicted, figsize=(5, 5))
    print("Show network weights matrix...")
    model.plot_weights()
    
if __name__ == '__main__':
    main()