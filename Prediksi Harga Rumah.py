#PREDIKSI HARGA RUMAH DI AREA SURAKARTA
"""
PERHATIAN!!
1. Karena jumlah independent variable (x) lebih dari 1 maka menggunakan Multiple Linear Regression
2. Rumusnya Y = b + e + m1*x1 + m2*x2 + … + mn*xn
3. Install terlebih dahulu semua library yang dibutuhkan, caranya buka cmd>pip instal ........
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

"""
4. Mengidentifikasi data
"""
# insert data csv menggunakan pandas
df = pd.read_csv('Harga Rumah Solo.csv', usecols=['bedrooms', 'bathrooms', 'building area', 'surface area', 'price']) #data diambil dari website 99.co pada tanggal 1 Juli 2020

#Independent variabel(x) adalah bedrooms, bathrooms, building area, surface are.
#Dependent variabel(y) adalah price.
print("\n===================================================================")
print("Berikut merupakan cuplikan dari DAFTAR HARGA RUMAH DI AREA SURAKARTA ")
print("===================================================================")
print(df.head())
print("\n===================================================================")

#Mengetahui jumlah kolom dan baris dari data
print("Jumlah Kolom dan Baris =", df.shape)

#Melihat informasi mulai dari jumlah data, tipe data, memory yang digunakan dll.
#Dapat dilihat bahwa seluruh data sudah di dalam bentuk numerik.
print(df.info())

#Melihat statistical description dari data mulai dari mean, kuartil, standard deviation dll.
print("\n=====================================================================")
print("Berikut data statistik dari 'DAFTAR HARGA RUMAH DI AREA SURAKARTA'")
print("=====================================================================")
print(df.describe(include='all'))
print("\n=====================================================================")

#Merubah tipe data dari bathrooms & bedrooms yang semula float (desimal) menjadi int
df['bedrooms'] = df['bedrooms'].astype('int')
df['bathrooms'] = df['bathrooms'].astype('int')

#Mencari missing values (mengecek apakah ada data yang kosong atau tidak.)
print("Pengecekan missing value")
print("=====================================================================")
print(df.isnull().sum())
print("\n=====================================================================")

"""
5. Exploratory Data Analysis untuk mengenal data lebih jauh.
"""
#Univariate analysis bedrooms.
f = plt.figure(figsize=(12,4))
f.add_subplot(1,2,1)
sns.countplot(df['bedrooms'])
f.add_subplot(1,2,2)
plt.boxplot(df['bedrooms'])

#Univariate analysis bathrooms.
f = plt.figure(figsize=(12,4))
f.add_subplot(1,2,1)
sns.countplot(df['bathrooms'])
f.add_subplot(1,2,2)
plt.boxplot(df['bathrooms'])

#Univariate analysis building area.
f = plt.figure(figsize=(12,4))
f.add_subplot(1,2,1)
df['building area'].plot(kind='kde')
f.add_subplot(1,2,2)
plt.boxplot(df['building area'])

#Univariate analysis surface area.
f = plt.figure(figsize=(12,4))
f.add_subplot(1,2,1)
df['surface area'].plot(kind='kde')
f.add_subplot(1,2,2)
plt.boxplot(df['surface area'])

#Bivariate analysis antara independent variable dan dependent variable.
sns.pairplot(data=df, x_vars=['bedrooms', 'bathrooms', 'building area', 'surface area'], y_vars=['price'], height=5, aspect=0.75)
plt.show()

"""
6. Membuat modeling data
"""
#Mengetahui nilai korelasi dari independent variable dan dependent variable.
print("Nilai Korelasi Antar Variable")
print("=====================================================================")
df.corr().style.background_gradient().set_precision(2)
print(df.corr())
print("\n=====================================================================")

##Pertama, buat variabel x dan y.
x = df.drop(columns='price')
y = df['price']

##Kedua, split data menjadi training and testing dengan porsi 80:20.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

##Ketiga, buat object linear regresi.
lin_reg = LinearRegression()

##Keempat, train the model menggunakan training data yang sudah displit.
lin_reg.fit(x_train, y_train)

##Kelima, cari tau nilai slope/koefisien (m) dan intercept (b).
print(lin_reg.coef_)
print(lin_reg.intercept_)

##Keenam, cari tahu accuracy score dari model menggunakan testing data yang sudah displit.
print("Akurasi skor modelnya adalah ", lin_reg.score(x_test, y_test))
print("\n=====================================================================")

"""
Melakukan prediksi harga rumah di area Surakarta
Misalnya Ruby ingin membeli rumah kriteria sebagai berikut:
1. Jumlah bedrooms = 5 buah
2. Jumlah bathrooms = 3 buah
3. Luas rumahnya = 250 m2
4. Luas tanahnya = 155 m2
"""
#Prediksi harga rumah

print("Prediksi harga rumah= Rp.", lin_reg.predict([[5,3,250,155]]))
print("\n=====================================================================")