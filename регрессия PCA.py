#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_excel(r"C:\Users\55944\Desktop\888\data_iqr_2.xlsx")


# In[3]:


x=df[['Соотношение матрица-наполнитель', 'Плотность, кг/м3',
       'модуль упругости, ГПа', 'Количество отвердителя, м.%',
       'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
       'Поверхностная плотность, г/м2', 'Модуль упругости при растяжении, ГПа',
       'Потребление смолы, г/м2',
       'Угол нашивки', 'Шаг нашивки', 'Плотность нашивки']]
# целевую переменную поместим в переменную y
y=df[['Прочность при растяжении, МПа']]


# In[4]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) 


# In[5]:


sc = StandardScaler() 
x_train = sc.fit_transform(x_train) 
x_test = sc.transform(x_test) 


# создаем объект PCA pca

# In[6]:


pca = PCA() 
x_train = pca.fit_transform(x_train) 
x_test = pca.transform(x_test) 


# Класс PCA содержит explained_variance_ratio_ которая возвращает дисперсию, вызванную каждым из основных компонентов. 
# 
#  

# In[7]:


explained_variance = pca.explained_variance_ratio_


# In[8]:


explained_variance 


#  первый главный компонент отвечает за дисперсию 9.8%. Точно так же второй главный компонент вызывает 9.5% отклонения в наборе данных,у всех 12 комонентов вкад в дисперсию  с разницей между макс. и мин. 3%.Таким образом 
#  метод показал значимость всех компонентов на одном уровне.
