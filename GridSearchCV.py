#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score,RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn import set_config
from time import time
import warnings
warnings.filterwarnings("ignore")


# Загружаем таблицу с локального сервера.Таблица очищена от выбросов,проверена на наличие пропусков и повторений,все данные имеют числовую характеристику. 

# In[2]:


df = pd.read_excel(r"C:\Users\55944\Desktop\888\data_iqr_2.xlsx")
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.head(5)


# In[3]:


columns = ['Соотношение матрица-наполнитель', 'Плотность, кг/м3',
       'модуль упругости, ГПа', 'Количество отвердителя, м.%',
       'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
       'Поверхностная плотность, г/м2', 'Модуль упругости при растяжении, ГПа','Потребление смолы', 'Угол нашивки',
       'Шаг нашивки', 'Плотность нашивки','Прочность при растяжении, МПа']


# In[4]:


df.columns


# In[5]:


#Нормализуем данные с помощью Normalizer,нормализация по столбцам
#создаем переменную 
norm_scaler=Normalizer()
df_norm=norm_scaler.fit_transform(np.array(df[['Соотношение матрица-наполнитель','Плотность, кг/м3','модуль упругости, ГПа','Количество отвердителя, м.%','Содержание эпоксидных групп,%_2','Температура вспышки, С_2','Поверхностная плотность, г/м2','Модуль упругости при растяжении, ГПа','Прочность при растяжении, МПа','Потребление смолы, г/м2','Угол нашивки','Шаг нашивки','Плотность нашивки']]))
df_norm[:1]


# In[6]:


#возвращаем названия столбцов,нормализованную матрицу
df_norm=pd.DataFrame(data=df_norm,columns=['Соотношение матрица-наполнитель','Плотность, кг/м3','модуль упругости, ГПа','Количество отвердителя, м.%','Содержание эпоксидных групп,%_2','Температура вспышки, С_2','Поверхностная плотность, г/м2','Модуль упругости при растяжении, ГПа','Прочность при растяжении, МПа','Потребление смолы','Угол нашивки','Шаг нашивки','Плотность нашивки'])
df_norm


# In[7]:


df_norm.columns


# In[8]:


# отберем признаки и поместим их в переменную X
x=df_norm[['Соотношение матрица-наполнитель', 'Плотность, кг/м3',
       'модуль упругости, ГПа', 'Количество отвердителя, м.%',
       'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
       'Поверхностная плотность, г/м2', 'Модуль упругости при растяжении, ГПа',
        'Потребление смолы', 'Угол нашивки',
       'Шаг нашивки', 'Плотность нашивки']]
# целевую переменную поместим в переменную y
y=df_norm[['Прочность при растяжении, МПа']]


# In[9]:


# разобьем данные на обучающую и тестовую выборку
# размер тестовой выборки составит 30%
# также зададим точку отсчета для воспроизводимости
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size = 0.3, 
                                                    random_state = 42)


# # Настройка гиперпараметров на одной модели – Регрессия

# Определяем модель

# In[10]:


regressor_GBR = GradientBoostingRegressor( random_state=0)
regressor_GBR.fit(x_train, y_train)


# In[11]:


y_pred_GBR_1 = regressor_GBR.predict(x_test)


# Смотрим результаты по умолчанию

# In[12]:


print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_GBR_1))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_GBR_1))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_GBR_1)))
print ('R Squared:', r2_score(y_test, y_pred_GBR_1))
print("Train score: {:.2f}".format(regressor_GBR.score(x_train, y_train)))
print("Test score: {:.2f}".format(regressor_GBR.score(x_test, y_test)))


# Метрики  регрессии показывают хорошие  результаты

# Прописываем параметры для нашей модели

# In[13]:


params = {'n_estimators':200,
          'max_depth':12,
          'criterion':'mse',
          'learning_rate':0.03,
          'min_samples_leaf':16,
          'min_samples_split':16
          }


# Тренируем

# In[14]:


gbr = GradientBoostingRegressor(**params)
gbr.fit(x_train,y_train)


# In[15]:


y_pred = gbr.predict(x_test)


# In[16]:


print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print ('R Squared:', r2_score(y_test, y_pred))
print("Train score: {:.2f}".format(regressor_GBR.score(x_train, y_train)))
print("Test score: {:.2f}".format(regressor_GBR.score(x_test, y_test)))


# Параметры оценки  по метрикам регрессии несколко ухудшились относительно модели с 
#  параметрами  по умолчанию.

# In[ ]:





# Определяем параметры  по которым будет обучаться модель и из списка которых будут
# рекомендоваться наилучшие.Поиск гиперепараметров.

# In[17]:


parameters = { 'loss' : ['ls', 'lad', 'huber', 'quantile'],
          'learning_rate' : (0.05,0.25,0.50,1),
          'criterion' : ['friedman_mse', 'mse', 'mae'],
          'max_features' : ['auto', 'sqrt', 'log2']
        }


# Подставляем параметры в модель и обучаем

# In[19]:


grid_1 = GridSearchCV(GradientBoostingRegressor(),parameters)
model_1 = grid_1.fit(x,y)
print(model_1.best_params_,'\n')
print(model_1.best_estimator_,'\n')


# Подставляем наилучшие параметры в выбранный алгоритм

# In[20]:


gbr_2 = GradientBoostingRegressor(criterion='mae', learning_rate=0.25, loss='ls',
                          max_features='auto')
gbr_2 .fit(x_train,y_train)


# In[21]:


y_pred_1 = gbr_2.predict(x_test)


# In[22]:


print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_1))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_1))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_1)))
print ('R Squared:', r2_score(y_test, y_pred_1))
print("Train score: {:.2f}".format(gbr_2.score(x_train, y_train)))
print("Test score: {:.2f}".format(gbr_2.score(x_test, y_test)))


# Оценки по метрикам регрессии изменились незначительно

# In[ ]:





# In[ ]:





# # Настройка гиперпараметров на нескольких моделях – Регрессия

# Выбор модели

# In[23]:


regressors = [
    KNeighborsRegressor(),
    SVR(),
    RidgeCV(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    LinearRegression(),
    LassoCV()
    ]


# In[24]:


cv = 10
for model in regressors[:cv]:
    start = time()
    model.fit(x_train, y_train)
    train_time = time() - start
    start = time()
    y_pred_1 = model.predict(x_test)
    predict_time = time()-start    
    print(model)
    print("\tTraining time: %0.3fs" % train_time)
    print("\tPrediction time: %0.3fs" % predict_time)
    print("\tExplained variance:", explained_variance_score(y_test, y_pred_1))
    print("\tMean Squared Error:", mean_squared_error(y_test, y_pred_1))
    print("\tRoot Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred_1)))
    print("\tMean absolute error:", mean_absolute_error(y_test, y_pred_1))
    print("\tR2 score:", r2_score(y_test, y_pred_1))
    print()


# Основная причина, по которой мы выполняем так много моделей вместе, заключается в том, чтобы определить, какая модель машинного обучения лучше всего подходит для данного конкретного набора данных.
# 
# Сравнивая эти результаты, мы можем быстро увидеть, что GradientBoostingRegressor является лучшей моделью для дальнейшего изучения. Но и другие модели показали хорошие результаты.Сравнение моделей лучше производить по метрике R2 score.
# 
# Предполагая, что GradientBoostingRegressor является лучшей моделью, теперь мы можем посмотреть, как выполняется настройка гиперпараметров на ней. Этот подход экономит время по сравнению с изучением каждой модели по отдельности.
# 

# ПОДБОР ГИПРЕПАРАМЕТРОВ

# In[25]:


parameters = { 'loss' : ['ls', 'lad', 'huber', 'quantile'],
          'learning_rate' : (0.05,0.25,0.50,1),
          'criterion' : ['friedman_mse', 'mse', 'mae'],
          'max_features' : ['auto', 'sqrt', 'log2']
        }


# In[26]:


grid = GridSearchCV(GradientBoostingRegressor(),parameters)
model = grid.fit(x,y)
print(model.best_params_,'\n')
print(model.best_estimator_,'\n')


# In[27]:


model.best_params_


# In[28]:


model.best_estimator_


# In[29]:


df_1 = pd.DataFrame(grid.cv_results_).set_index('rank_test_score').sort_index()
df_1.shape


# In[30]:


df_1


# In[31]:


regressor_GBR_1 = GradientBoostingRegressor( criterion='mae', learning_rate=0.25, loss='ls',
                                           max_features='auto',random_state=0)
regressor_GBR_1.fit(x_train, y_train)


# In[32]:


y_pred_2 =regressor_GBR_1.predict(x_test)


# In[33]:


print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_2))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_2))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_2)))
print ('R Squared:', r2_score(y_test, y_pred_2))
print("Train score: {:.2f}".format(regressor_GBR_1.score(x_train, y_train)))
print("Test score: {:.2f}".format(regressor_GBR_1.score(x_test, y_test)))


# Метод   GridSearchCV позволяет нахотить наилучший метод с наилучшими гиперпараметами
# и весь поиск поисходит в автоматическом режиме.Вданном случае нам не намного удалось
# удалось улучшить алгоритм GradientBoostingRegressor относительно его применения
# с параметрами о умолчанию.

# In[ ]:




