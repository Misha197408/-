#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
import scipy.stats as stats


# # обработка датасета и разведочный анализ данных

# In[2]:


#Загружаем и обрабатываем входящие датасеты
#Загружаем  датасет (базальтопластик) и посмотрим на названия столбцов
df_bp = pd.read_excel(r'C:\Users\55944\Desktop\888\X_bp.xlsx')
df_bp.shape


# In[3]:


#Смотрим таблицу
df_bp.head(5)


# In[4]:


#Удаляем первый неинформативный столбец
df_bp.drop(['Unnamed: 0'], axis=1, inplace=True)
df_bp.head(5)


# In[5]:


df_bp.shape


# In[6]:


# Загружаем второй датасет  
df_nup = pd.read_excel(r'C:\Users\55944\Desktop\888\X_nup.xlsx')
df_nup.shape


# In[7]:


#Посмотрим на первые 5 строк второго датасета
df_nup.head()


# In[8]:


#Удаляем первый неинформативный столбец
df_nup.drop(['Unnamed: 0'], axis=1, inplace=True)


# In[9]:


#два датасета имеют разный объем строк.  
# По условию задачи объединяем их по типу INNER
#объединяем таблицы
df = pd.concat([df_bp, df_nup], axis=1, join="inner")
df.head().T


# In[10]:


df.shape#смотрим колличество столбцов и строк.объединение произошло с удалением лишних строк.


# In[11]:


#проведем анализ табпицы
df.info()


# In[12]:


#все значения в таблице имеются,пустых значений нет, у всех значений числовой характер.
#найдем уникальные значения.
df.nunique()


# In[13]:


df['Угол нашивки, град'][df['Угол нашивки, град'] == 0].count()


# In[14]:


df['Угол нашивки, град'][df['Угол нашивки, град'] == 90].count()


# In[15]:


df.describe()


# In[16]:


#медиана показывает центральное значение в выборке,если наблюдений нечетное колличество
#и седнее аифметическое ,если четное.У нас нечетное колличество 1023 наблюдений.
#в столбце угол нашивки два значения в градусах 0 и 90.
#приведем значение 0 град. к числу 0,а 90 град.к числу 1.
df=df.replace({'Угол нашивки, град':{0:0,90:1}})


# In[17]:


# Приведем столбец "Угол нашивки" к  float

df['Угол нашивки, град'] = df['Угол нашивки, град'].astype(float)


# In[18]:


df.head()


# In[19]:


df.info()


# In[20]:


#поменяем название столбца,уберем градусы
df = df.rename(columns={'Угол нашивки, град' : 'Угол нашивки'})
df


# In[21]:


df.describe()


# # Описательная статистика содержит по каждому столбцу (по каждой переменной):
# 
#     count - количество значений
#     mean - среднее значение
#     std - стандартное отклонение
#     min - минимум
#     25% - верхнее значение первого квартиля
#     50% - медиана
#     75% - верхнее значение третьего квартиля
#     max - максимум
# 
# 

# In[22]:


#проверим таблицу на пропущенные данные
df.isnull().sum()


# In[23]:


cols = df.columns 
# определяем цвета 
# желтый -не пропущенные данные, синий -  пропущенные
colours = ['#000099', '#ffff00'] 
sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))


# In[24]:


#проверим таблицу на наличие дублирующих значений
df.duplicated().sum()


# In[25]:


#получим средние и медианные значения
df.mean()


# In[26]:


df.median()


# # Выявляем зависимость между переменными. 

# In[27]:


# Вычисляем коэффициенты ранговой корреляции Кендалла. Статистической зависимости не наблюдаем.
df.corr(method = 'kendall')


# In[28]:


#Вычисляем коэффициенты корреляции Пирсона. Статистической зависимости не наблюдаем.
df.corr(method ='pearson')


# In[29]:


#все зависимости близки к нулевым значениям,зависимости не наблюдается.


# In[30]:


sns.heatmap(df.corr(method ='pearson'))


# In[31]:


sns.heatmap(df.corr(method = 'kendall'))


# In[32]:


#Визуализация корреляционной матрицы с помощью тепловой карты
mask = np.triu(df.corr())
# Создаем полотно для отображения большого графика
f, ax = plt.subplots(figsize = (15, 10))
# # Визуализируем данные кореляции и создаем цветовую палитру
sns.heatmap(df.corr(method = 'kendall'), mask = mask, annot = True, square = True, cmap = 'coolwarm')
plt.xticks(rotation =90 , ha='right')
plt.show()


# In[33]:


#Визуализация корреляционной матрицы с помощью тепловой карты
mask = np.triu(df.corr())
# Создаем полотно для отображения большого графика
f, ax = plt.subplots(figsize = (15, 10))
# # Визуализируем данные кореляции и создаем цветовую палитру
sns.heatmap(df.corr(method ='pearson'), mask = mask, annot = True, square = True, cmap = 'coolwarm')
plt.xticks(rotation =90 , ha='right')
plt.show()


# Судя по визуализации теловых карт корреляции двумя разными методами,
# корреляции между признаками отсутствует.

# In[34]:


g = sns.PairGrid(df[df.columns])# Попарные графики рассеяния точек
g.map(sns.scatterplot, color = 'brown')
g.map_upper(sns.scatterplot, color = 'green')
g.map_lower(sns.kdeplot, color = 'blue')
plt.show


# По данной визуализации ,так же можно утверждать ,что зависимости между 
# переменными,судя о форме распределения точек отсутствуют.

# # Определяем характер распределения переменных относительно нормального распределения
# 

# In[35]:


#строим гистограммы
df.hist(figsize = (15,15), color = "r")
plt.show()


# Все гистограммы близки к нормальному распределению,исключение угол нашивки,
# пример распределения дискретной величины.

# # Исследуем данные на наличие выбросов и аномалий

# In[36]:


# Ящики с усами 
a = 5 # количество строк
b = 5 # количество столцбцов
c = 1 # инициализация plot counter

plt.figure(figsize = (35,35))
plt.suptitle('Диаграммы "ящики с усами"', y = 0.9 ,
             fontsize = 30)
for col in df.columns:
    plt.subplot(a, b, c)
    #plt.figure(figsize=(7,5))
    sns.boxplot(data = df, y = df[col], fliersize = 15, linewidth = 5, boxprops = dict(facecolor = 'y', color = 'g'), medianprops = dict(color = 'lime'), whiskerprops = dict(color="g"), capprops = dict(color = "yellow"), flierprops = dict(color="y", markeredgecolor = "lime"))
    plt.ylabel(None)
    plt.title(col, size = 20)
    #plt.show()
    c += 1
# "Ящики с усами" показывают наличие выбросов во всех столбцах, кроме углов нашивки


# In[37]:


# гистограмма распределения и боксплоты 
for column in df.columns:
    fig= px.histogram(df, x = column, color_discrete_sequence = ['blue'], nbins = 200, marginal = "box")
    fig.show()


# Данный боксплот интерактивный.Показывает координаты выбросов при наведении
# курсора.

# # удаление выбросов

# Метод межквартильного диапазона

# In[38]:


Q1 = df.quantile(q=.25)
Q3 = df.quantile(q=.75)
IQR = df.apply(stats.iqr)
data_iqr = df[~((df < (Q1-1.5*IQR)) | (df > (Q3+1.5*IQR))).any(axis=1)]


# смотрим размерность таблицы,как она изменилась.

# In[39]:


data_iqr.shape


# смотрим на боксплот

# In[40]:


for column in data_iqr.columns:
    fig= px.histogram(data_iqr, x = column, color_discrete_sequence = ['blue'], nbins = 200, marginal = "box")
    fig.show()


# модуль упругости при растяжении,прочность при растяжении,потребление смолы
# плотность нашивки наблюдаются выбросы.

# Метод межквартильного диапазона,применяем повторно

# In[41]:


Q1 = data_iqr.quantile(q=.25)
Q3 = data_iqr.quantile(q=.75)
IQR = data_iqr.apply(stats.iqr)
data_iqr_1 = data_iqr[~((data_iqr < (Q1-1.5*IQR)) | (data_iqr > (Q3+1.5*IQR))).any(axis=1)]
data_iqr_1.shape


# Проверяем выбросы

# In[42]:


for column in data_iqr_1.columns:
    fig= px.histogram(data_iqr_1, x = column, color_discrete_sequence = ['blue'], nbins = 200, marginal = "box")
    fig.show()


# прочность при ррастяжении,плотность нашивки имеют выбросы

# Метод межквартильного диапазона,применяем последовательно в третий раз

# In[43]:


Q1 = data_iqr_1.quantile(q=0.25)
Q3 = data_iqr_1.quantile(q=0.75)
IQR = data_iqr_1.apply(stats.iqr)
data_iqr_2 = data_iqr_1[~((data_iqr_1<(Q1-1.5*IQR))|(data_iqr_1>(Q3+1.5*IQR))).any(axis=1)]
data_iqr_2.shape


# Проверяем выбросы

# In[44]:


for column in data_iqr_2.columns:
    fig= px.histogram(data_iqr_2, x = column, color_discrete_sequence = ['blue'], nbins = 200, marginal = "box")
    fig.show()


# Выбросы не наблюдаются.Удалось очистить данные от выбросов после трех 
# последовательных применений метода межквартильного диапазона

# In[45]:


data_iqr_2


# In[46]:


# Посмотрим на средние и медианные знчения датасета до удаления выбросов 
mean_and_50 = df.describe()
mean_and_50.loc[['mean', '50%']]


# In[47]:


# Посмотрим на средние и медианные знчения датасета после удаления выбросов 
mean_and_50 = data_iqr_2.describe()
mean_and_50.loc[['mean', '50%']]


# после удаления выбросов,колличество строк изменилось.Было 1023 стало 922,измнилась характеистика 
# числа с нечетного на четное.Поэтому кардинально изменилось медианное значение столбца 'Угол нашивки',
# с 0 на 1.Все остальные значения поменялись крайне несущественно.

# In[48]:


#data_iqr_2.to_excel(r"C:\Users\55944\Desktop\888\data_iqr_2.xlsx")#сохраняем таблицу

