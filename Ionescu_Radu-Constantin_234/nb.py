from PIL import Image
import numpy as np
import os
from sklearn.naive_bayes import MultinomialNB
import csv
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# M-am gandit ca daca as dubla datele de antrenare adaugand imaginea si versiunea ei sepia acuratetea ar creste
# deoarece avem mai multe date de antrenare. Practic, acuratatea e putin mai ridicata (aproximativ cu 5%)
# cand NU folosesc dublarea de date cu sepia si in plus modelul se compileaza mai repede
"""
def make_sepia(img):
    pixels = list(img.getdata())
    sepia_pixels = []
    for pixel in pixels:
        r, g, b = pixel
        new_r = min(int(r * 0.393 + g * 0.769 + b * 0.189), 255)
        new_g = min(int(r * 0.349 + g * 0.686 + b * 0.168), 255)
        new_b = min(int(r * 0.272 + g * 0.534 + b * 0.131), 255)
        sepia_pixels.append((new_r, new_g, new_b))
    sepia_img = Image.new("RGB", img.size)
    sepia_img.putdata(sepia_pixels)
    return np.array(sepia_img)
"""

# Prima zona comuna pentru ambele modele: inputul
# transformam imaginea din fisier PNG in numpy array (de dimensiune 64 x 64 x 3)
def load_images(folder):
    images = dict()
    for filename in os.listdir(folder):
        with Image.open(os.path.join(folder, filename)) as img:
            images[filename] = np.array(img)
    return images

# Citirea datelor de antrenare, validare si testare (imi pare rau ca este un cod incurcat,
# m-am chinuit sa fac cumva citirea imaginilor si e posibil sa ma fi complicat cu dictionarele dar dupa
# ce am vazut ca functioneaza nu am mai rafinat citirea)

train_data_dict = load_images('/kaggle/input/unibuc-dhc-2023/train_images')
test_images_dict = load_images('/kaggle/input/unibuc-dhc-2023/test_images')
val_data_dict = load_images('/kaggle/input/unibuc-dhc-2023/val_images')

# Dictionare pentru a retine asocierile imagine - label
train_labels_dict = {line.split(',')[0]: int(line.split(',')[1].strip('\n')) for line in
                     open("/kaggle/input/unibuc-dhc-2023/train.csv", "r").readlines()[1:]}
test_images_names = [line.strip('\n') for line in open("/kaggle/input/unibuc-dhc-2023/test.csv", "r").readlines()[1:]]
val_labels_dict = {line.split(',')[0]: int(line.split(',')[1].strip('\n')) for line in
                   open("/kaggle/input/unibuc-dhc-2023/val.csv", "r").readlines()[1:]}

# Aducem datele de input la forma de input numpy array studiata la laborator
train_images = []
train_labels = []

for nume_imagine in train_data_dict.keys():
    train_images.append(train_data_dict[nume_imagine])
    train_labels.append(train_labels_dict[nume_imagine])

val_images = []
val_labels = []

for nume_imagine in val_data_dict.keys():
    val_images.append(val_data_dict[nume_imagine])
    val_labels.append(val_labels_dict[nume_imagine])

train_images = np.array(train_images)
val_images = np.array(val_images)
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
train_label_counter = Counter(train_labels)
val_label_counter = Counter(val_labels)
# Folosim to_categorical pentru ca problema este una de clasificare in categorii
train_labels = to_categorical(train_labels, num_classes=96)
val_labels = to_categorical(val_labels, num_classes=96)

# Reshape cu -1: lasam numpy sa calculeze automat dimensiunea datelor
train_images = np.array(train_images).reshape(len(train_images), -1)
val_images = np.array(val_images).reshape(len(val_images), -1)

test_images = np.array([test_images_dict[img] for img in test_images_names])
test_images = np.array(test_images).reshape(len(test_images), -1)

# Vizualizare date de antrenare si validare
plt.figure(figsize=(15, 5))
classes_train = list(train_label_counter.keys())
counts_train = list(train_label_counter.values())

plt.bar(classes_train, counts_train)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Train Data Distribution')
plt.xticks(classes_train, rotation='vertical')
plt.show()

plt.figure(figsize=(15, 5))
classes_val = list(val_label_counter.keys())
counts_val = list(val_label_counter.values())

plt.bar(classes_val, counts_val)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Validation Data Distribution')
plt.xticks(classes_val, rotation='vertical')
plt.show()

#############################################
# Rezolvare cu Naive Bayes ca in laboratorul 2


# Impartim valorile continue in containere discrete
def values_to_bins(x, bins):
    return np.digitize(x, bins) - 1

# MultinomialNB primeste ca parametru un array unidimensional, asa ca aducem etichetele la aceasta forma
train_labels = np.argmax(train_labels, axis=1)
val_labels = np.argmax(val_labels, axis=1)

# Verificam pentru mai multe modele in functie de numarul de containere ales
# care este cel optim pe datele de validare in functca apoi sa il fololsim
d_acc = dict()

# lista de impartiri pentru containere de la laborator dar extinsa pana la 27
for num_bins in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]:
    bins = np.linspace(0, 255, num=num_bins)
    x_train = values_to_bins(train_images, bins)
    x_val = values_to_bins(val_images, bins)
    clf = MultinomialNB()
    clf.fit(x_train, train_labels)
    accuracy = clf.score(x_val, val_labels)
    print('Accuracy pentru ' + str(num_bins) + ' bins: ', accuracy)
    d_acc[num_bins] = accuracy

ideal_number_of_bins = max(d_acc, key=d_acc.get)
print(ideal_number_of_bins)

bins = np.linspace(0, 255, num=ideal_number_of_bins)
x_train = values_to_bins(train_images, bins)
x_test = values_to_bins(test_images, bins)

clf = MultinomialNB()
clf.fit(x_train, train_labels)
predicted_labels = clf.predict(x_test)
val_preds = clf.predict(val_images)

#############################################
# A doua zona comuna pentru modele: outputul
# Generarea fisierului de output pentru trimis la submsissions
output = list(zip(test_images_names, predicted_labels))
df = pd.DataFrame(output)
df.to_csv('output.csv', index=False,  header=['Image', 'Class'])

# Folosim modelul pe datele de validare pentru a obtine matricea de confuzie
val_preds_classes = val_preds
val_true_classes = val_labels
conf_matrix = confusion_matrix(val_true_classes, val_preds_classes)

# Calculam acuratetea generala si acuratetea pe clase
pred_labels = list(val_preds_classes)
true_labels = list(val_true_classes)
# calculam acuratetea generala ca fiind suma elementelor de pe diagonala principala supra numarul total de elemente din matricea de confuzie
# calculam acuratetea per clasa ca fiind numarul de elemente corect clasificate (conf_matrix[i, i])
# supra numarul real total de elemente din clasa respectiva (true_labels.count(i))
total_accuracy = sum([conf_matrix[i, i] for i in range(96)]) / np.sum(conf_matrix)
per_classes_accuracy = np.array([conf_matrix[i, i] / true_labels.count(i) for i in range(96)]) * 100

# Calculul preciziei si recall-ului pentru fiecare clasa
per_classes_precision = np.array([conf_matrix[i, i] / sum(conf_matrix[:, i]) for i in range(96)]) * 100 # suma elementelor de pe toata coloana reprezinta toate clasificarile pozitive, si corecte si gresite
per_classes_recall = np.array([conf_matrix[i, i] / sum(conf_matrix[i, :]) for i in range(96)]) * 100 # suma elementelor de pe toata linia reprezinta toate clasificarile pozitive corecte si cele negative gresite


# Vizualizarea pentru matricea de confuzie
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, annot=False, cmap='viridis')
plt.title('Confusion matrix - ' + str(total_accuracy * 100) + '% accuracy')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# Vizualizarea pentru acuratetea per clasa
plt.figure(figsize=(15, 5))
classes = np.arange(len(per_classes_accuracy))
plt.bar(classes, per_classes_accuracy)
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.title('Per-Class Accuracy')
plt.xticks(classes, rotation='vertical')
plt.show()

# Vizualizarea pentru precizia per clasa
plt.figure(figsize=(15, 5))
classes = np.arange(len(per_classes_precision))
plt.bar(classes, per_classes_precision)
plt.xlabel('Class')
plt.ylabel('Precision (%)')
plt.title('Per-Class Precision')
plt.xticks(classes, rotation='vertical')
plt.show()

# Vizualizarea pentru recall-ul per clasa
plt.figure(figsize=(15, 5))
classes = np.arange(len(per_classes_recall))
plt.bar(classes, per_classes_recall)
plt.xlabel('Class')
plt.ylabel('Recall (%)')
plt.title('Per-Class Recall')
plt.xticks(classes, rotation='vertical')
plt.show()