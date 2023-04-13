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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
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


# Проводим масштабирование переменных используя Normalizer

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


# Выделяем признаки и целевую переменную.Разбиваем данные на тренировочную и    тестовую выборки.

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


x.columns


# In[10]:


print(type(x), type(y))


# In[11]:


# разобьем данные на обучающую и тестовую выборку
# размер тестовой выборки составит 30%
# также зададим точку отсчета для воспроизводимости
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size = 0.3, 
                                                    random_state = 42)


# In[ ]:





# # ПРОГНОЗИРУЕМ ПРОЧНОСТЬ ПРИ РАСТЯЖЕНИИ

# # Регрессия случайного леса
# ​

# In[12]:


regressor_RF = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor_RF.fit(x_train, y_train) 


# In[13]:


y_pred_RF = regressor_RF.predict(x_test)


# In[14]:


print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_RF))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_RF))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_RF)))
print ('R_Squared:', r2_score(y_test, y_pred_RF))


# In[15]:


R_Squared_RF=r2_score(y_test, y_pred_RF)


# In[16]:


print("Train score: {:.2f}".format(regressor_RF.score(x_train, y_train)))
print("Test score: {:.2f}".format(regressor_RF.score(x_test, y_test)))


# In[17]:


plt.figure(figsize = (10, 7))
plt.title("Тестовые и прогнозные значения RandomForestRegressor")
plt.plot(y_pred_RF, label = "Прогноз", color = 'red')
plt.plot(y_test.values, label = "Тест", color = 'darkgreen')
plt.xlabel("Количество наблюдений")
plt.ylabel("Модуль упругости при растяжении, ГПа")
plt.legend()
plt.grid(True);


# # Регрессия дерева решений

# In[18]:


from graphviz import Source
from sklearn import tree


# In[19]:


regressor_DT = DecisionTreeRegressor( random_state = 0)
regressor_DT.fit(x_train, y_train)


# In[20]:


tree.plot_tree(regressor_DT)


# In[21]:


y_pred_DT = regressor_DT.predict(x_test)


# In[22]:


print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_DT))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_DT))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_DT)))
print ('R Squared:', r2_score(y_test, y_pred_DT))


# In[23]:


R_Squared_DT=r2_score(y_test, y_pred_DT)


# In[24]:


print("Train score: {:.2f}".format(regressor_DT.score(x_train, y_train)))
print("Test score: {:.2f}".format(regressor_DT.score(x_test, y_test)))


# In[25]:


plt.figure(figsize = (10, 7))
plt.title("Тестовые и прогнозные значения Decision Tree Regressor")
plt.plot(y_pred_DT, label = "Прогноз", color = 'red')
plt.plot(y_test.values, label = "Тест", color = 'darkgreen')
plt.xlabel("Количество наблюдений")
plt.ylabel("Модуль упругости при растяжении, ГПа")
plt.legend()
plt.grid(True);


# # Регрессия LASSO

# In[26]:


from sklearn import linear_model
regressor_LS = linear_model.Lasso(alpha=0.1)
regressor_LS.fit(x_train, y_train)


# In[27]:


y_pred_LS = regressor_LS.predict(x_test)


# In[28]:


print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_LS))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_LS))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_LS)))
print ('R Squared:', r2_score(y_test, y_pred_LS))


# In[29]:


R_Squared_LS=r2_score(y_test, y_pred_LS)


# In[30]:


print("Train score: {:.2f}".format(regressor_LS.score(x_train, y_train)))
print("Test score: {:.2f}".format(regressor_LS.score(x_test, y_test)))


# In[31]:


plt.figure(figsize = (10, 7))
plt.title("Тестовые и прогнозные значения LASSO")
plt.plot(y_pred_LS, label = "Прогноз", color = 'red')
plt.plot(y_test.values, label = "Тест", color = 'darkgreen')
plt.xlabel("Количество наблюдений")
plt.ylabel("Модуль упругости при растяжении, ГПа")
plt.legend()
plt.grid(True);


# In[ ]:





# # kNN

# In[32]:


regressor_KNN = KNeighborsRegressor(n_neighbors=3)
regressor_KNN.fit(x_train, y_train)


# In[33]:


y_pred_KNN = regressor_KNN.predict(x_test)


# In[34]:


print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_KNN))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_KNN))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_KNN)))
print ('R Squared:', r2_score(y_test, y_pred_KNN))


# In[35]:


R_Squared_KNN=r2_score(y_test, y_pred_KNN)


# In[36]:


print("Train score: {:.2f}".format(regressor_KNN.score(x_train, y_train)))
print("Test score: {:.2f}".format(regressor_KNN.score(x_test, y_test)))


# In[37]:


plt.figure(figsize = (10, 7))
plt.title("Тестовые и прогнозные значения KNeighborsRegressor")
plt.plot(y_pred_DT, label = "Прогноз", color = 'red')
plt.plot(y_test.values, label = "Тест", color = 'darkgreen')
plt.xlabel("Количество наблюдений")
plt.ylabel("Модуль упругости при растяжении, ГПа")
plt.legend()
plt.grid(True);


# # Метод опорных векторов(Support Vector Machines)
# 

# In[38]:


from sklearn import svm
regressor_SVR = svm.SVR()
regressor_SVR.fit(x_train, y_train)


# In[39]:


y_pred_SVR = regressor_SVR.predict(x_test)


# In[40]:


print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_SVR))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_SVR))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_SVR)))
print ('R Squared:', r2_score(y_test, y_pred_SVR))


# In[41]:


R_Squared_SVR=r2_score(y_test, y_pred_SVR)


# In[42]:


print("Train score: {:.2f}".format(regressor_SVR.score(x_train, y_train)))
print("Test score: {:.2f}".format(regressor_SVR.score(x_test, y_test)))


# In[43]:


plt.figure(figsize = (10, 7))
plt.title("Тестовые и прогнозные значения Support Vector Machines")
plt.plot(y_pred_SVR, label = "Прогноз", color = 'red')
plt.plot(y_test.values, label = "Тест", color = 'darkgreen')
plt.xlabel("Количество наблюдений")
plt.ylabel("Модуль упругости при растяжении, ГПа")
plt.legend()
plt.grid(True);


# # Гребневая регрессия (ридж-регрессия)

# In[44]:


regressor_R = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
regressor_R.fit(x_train, y_train)


# In[45]:


y_pred_R = regressor_R.predict(x_test)


# In[46]:


print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_R))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_R))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_R)))
print ('R Squared:', r2_score(y_test, y_pred_R))


# In[47]:


R_Squared_R=r2_score(y_test, y_pred_R)


# In[48]:


print("Train score: {:.2f}".format(regressor_R.score(x_train, y_train)))
print("Test score: {:.2f}".format(regressor_R.score(x_test, y_test)))


# In[49]:


plt.figure(figsize = (10, 7))
plt.title("Тестовые и прогнозные значения RidgeCV")
plt.plot(y_pred_DT, label = "Прогноз", color = 'red')
plt.plot(y_test.values, label = "Тест", color = 'darkgreen')
plt.xlabel("Количество наблюдений")
plt.ylabel("Модуль упругости при растяжении, ГПа")
plt.legend()
plt.grid(True);


# # Метод градиентного бустинга - Gradient Boosting Regressor

# In[50]:


regressor_GBR = GradientBoostingRegressor( random_state=0)
regressor_GBR.fit(x_train, y_train)


# In[51]:


y_pred_GBR = regressor_GBR.predict(x_test)


# In[52]:


print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_GBR))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_GBR))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_GBR)))
print ('R Squared:', r2_score(y_test, y_pred_GBR))


# In[53]:


R_Squared_GBR=r2_score(y_test, y_pred_GBR)


# In[54]:


print("Train score: {:.2f}".format(regressor_GBR.score(x_train, y_train)))
print("Test score: {:.2f}".format(regressor_GBR.score(x_test, y_test)))


# In[55]:


plt.figure(figsize = (10, 7))
plt.title("Тестовые и прогнозные значения GradientBoostingRegressor")
plt.plot(y_pred_DT, label = "Прогноз", color = 'red')
plt.plot(y_test.values, label = "Тест", color = 'darkgreen')
plt.xlabel("Количество наблюдений")
plt.ylabel("Модуль упругости при растяжении, ГПа")
plt.legend()
plt.grid(True);


# Создадим таблицу с значениями R_Squared для каждого Регрессора

# In[56]:


R_Squared_df_norm = {'Регрессор': ['Support Vector', 'RandomForest',  'GradientBoosting', 'KNeighbors', 'DecisionTree','Lasso','Ridge',], 'R_Squared': [R_Squared_SVR,R_Squared_RF,R_Squared_GBR,R_Squared_KNN, R_Squared_DT, R_Squared_LS,R_Squared_R]}
R_Squared_df_norm = pd.DataFrame(R_Squared_df_norm)


# In[57]:


R_Squared_df_norm


# Наилучший показатель по метрике-R_Squared для сравнения работы алгоритмов -GradientBoosting,
# наихудший -Lasso.

# # Прогнозируем модуль упругости при растяжении

# In[58]:


df_norm.columns


# In[59]:


# отберем признаки и поместим их в переменную X
x_2=df_norm[['Соотношение матрица-наполнитель', 'Плотность, кг/м3',
       'модуль упругости, ГПа', 'Количество отвердителя, м.%',
       'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
       'Поверхностная плотность, г/м2', 
       'Прочность при растяжении, МПа', 'Потребление смолы', 'Угол нашивки',
       'Шаг нашивки', 'Плотность нашивки']]
# целевую переменную поместим в переменную y
y_2=df_norm[['Модуль упругости при растяжении, ГПа']]


# In[60]:


# разобьем данные на обучающую и тестовую выборку
# размер тестовой выборки составит 30%
# также зададим точку отсчета для воспроизводимости
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_2, y_2, 
                                                    test_size = 0.3, 
                                                    random_state = 42)


# In[61]:


# Проверка правильности разбивки
df_norm.shape[0] - x_train_2.shape[0] - x_test_2.shape[0]


# In[62]:


y_train_2.shape


# In[63]:


x_train_2.shape


# In[64]:


x_train_2.head()


# In[65]:


y_2.head()


# In[66]:


x_2.head()


# ПОВТОРЯЕМ ВСЕ ДЕЙСТВИЯ ПО НАПИСАНИЮ АЛГОРИТМОВ,ПРОИЗВЕДЕННЫХ С ПРОГНОЗИРОВАНИЕМ 
# ПРОЧНОСТИ ПРИ РАСТЯЖЕНИИ,ТОЛЬКО С ДРУГОЙ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ-модуль упругости при растяжении

# # Регрессия случайного леса

# In[67]:


regressor_RF_2 = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor_RF_2.fit(x_train_2, y_train_2) 


# In[68]:


y_pred_RF_2 = regressor_RF_2.predict(x_test_2)


# In[69]:


print('Mean Absolute Error:', mean_absolute_error(y_test_2, y_pred_RF_2))
print('Mean Squared Error:', mean_squared_error(y_test_2, y_pred_RF_2))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_2, y_pred_RF_2)))
print ('R_Squared:', r2_score(y_test_2, y_pred_RF_2))


# In[70]:


R_Squared_RF_2=r2_score(y_test_2, y_pred_RF_2)


# In[71]:


print("Train score: {:.2f}".format(regressor_RF_2.score(x_train_2, y_train_2)))
print("Test score: {:.2f}".format(regressor_RF_2.score(x_test_2, y_test_2)))


# In[72]:


plt.figure(figsize = (10, 7))
plt.title("Тестовые и прогнозные значения RandomForestRegressor")
plt.plot(y_pred_DT, label = "Прогноз", color = 'red')
plt.plot(y_test.values, label = "Тест", color = 'darkgreen')
plt.xlabel("Количество наблюдений")
plt.ylabel("Модуль упругости при растяжении, ГПа")
plt.legend()
plt.grid(True);


# # Регрессия дерева решений

# In[73]:


regressor_DT_2 = DecisionTreeRegressor( random_state = 0)
regressor_DT_2.fit(x_train_2, y_train_2)


# In[74]:


y_pred_DT_2 = regressor_DT_2.predict(x_test_2)


# In[75]:


print('Mean Absolute Error:', mean_absolute_error(y_test_2, y_pred_DT_2))
print('Mean Squared Error:', mean_squared_error(y_test_2, y_pred_DT_2))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_2, y_pred_DT_2)))
print ('R_Squared:', r2_score(y_test_2, y_pred_DT_2))


# In[76]:


R_Squared_DT_2=r2_score(y_test_2, y_pred_DT_2)


# In[77]:


print("Train score: {:.2f}".format(regressor_DT_2.score(x_train_2, y_train_2)))
print("Test score: {:.2f}".format(regressor_DT_2.score(x_test_2, y_test_2)))


# In[78]:


tree.plot_tree(regressor_DT_2)


# In[79]:


plt.figure(figsize = (10, 7))
plt.title("Тестовые и прогнозные значения DecisionTreeRegressor")
plt.plot(y_pred_DT, label = "Прогноз", color = 'red')
plt.plot(y_test.values, label = "Тест", color = 'darkgreen')
plt.xlabel("Количество наблюдений")
plt.ylabel("Модуль упругости при растяжении, ГПа")
plt.legend()
plt.grid(True);


# # Регрессия LASSO

# In[80]:


from sklearn import linear_model
regressor_LS_2 = linear_model.Lasso(alpha=0.1)
regressor_LS_2.fit(x_train_2, y_train_2)


# In[81]:


y_pred_LS_2 = regressor_LS_2.predict(x_test_2)


# In[82]:


print('Mean Absolute Error:', mean_absolute_error(y_test_2, y_pred_LS_2))
print('Mean Squared Error:', mean_squared_error(y_test_2, y_pred_LS_2))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_2, y_pred_LS_2)))
print ('R Squared:', r2_score(y_test_2, y_pred_LS_2))


# In[83]:


R_Squared_LS_2=r2_score(y_test_2, y_pred_LS_2)


# In[84]:


print("Train score: {:.2f}".format(regressor_LS_2.score(x_train_2, y_train_2)))
print("Test score: {:.2f}".format(regressor_LS_2.score(x_test_2, y_test_2)))


# In[85]:


plt.figure(figsize = (10, 7))
plt.title("Тестовые и прогнозные значения Lasso")
plt.plot(y_pred_DT, label = "Прогноз", color = 'red')
plt.plot(y_test.values, label = "Тест", color = 'darkgreen')
plt.xlabel("Количество наблюдений")
plt.ylabel("Модуль упругости при растяжении, ГПа")
plt.legend()
plt.grid(True);


# # kNN

# In[86]:


regressor_KNN_2 = KNeighborsRegressor(n_neighbors=3)
regressor_KNN_2.fit(x_train_2, y_train_2)


# In[87]:


y_pred_KNN_2 = regressor_KNN.predict(x_test_2)


# In[88]:


print('Mean Absolute Error:', mean_absolute_error(y_test_2, y_pred_KNN_2))
print('Mean Squared Error:', mean_squared_error(y_test_2, y_pred_KNN_2))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_2, y_pred_KNN_2)))
print ('R Squared:', r2_score(y_test_2, y_pred_KNN_2))


# In[89]:


R_Squared_KNN_2=r2_score(y_test_2, y_pred_KNN_2)


# In[90]:


print("Train score: {:.2f}".format(regressor_KNN_2.score(x_train_2, y_train_2)))
print("Test score: {:.2f}".format(regressor_KNN_2.score(x_test_2, y_test_2)))


# In[91]:


plt.figure(figsize = (10, 7))
plt.title("Тестовые и прогнозные значения KNeighborsRegressor")
plt.plot(y_pred_DT, label = "Прогноз", color = 'red')
plt.plot(y_test.values, label = "Тест", color = 'darkgreen')
plt.xlabel("Количество наблюдений")
plt.ylabel("Модуль упругости при растяжении, ГПа")
plt.legend()
plt.grid(True);


# # Метод опорных векторов(Support Vector Machines)

# In[92]:


from sklearn import svm
regressor_SVR_2 = svm.SVR()
regressor_SVR_2.fit(x_train_2, y_train_2)


# In[93]:


y_pred_SVR_2 = regressor_SVR_2.predict(x_test_2)


# In[94]:


print('Mean Absolute Error:', mean_absolute_error(y_test_2, y_pred_SVR_2))
print('Mean Squared Error:', mean_squared_error(y_test_2, y_pred_SVR_2))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_2, y_pred_SVR_2)))
print ('R Squared:', r2_score(y_test_2, y_pred_SVR_2))


# In[95]:


R_Squared_SVR_2=r2_score(y_test_2, y_pred_SVR_2)


# In[96]:


print("Train score: {:.2f}".format(regressor_SVR_2.score(x_train_2, y_train_2)))
print("Test score: {:.2f}".format(regressor_SVR_2.score(x_test_2, y_test_2)))


# In[97]:


plt.figure(figsize = (10, 7))
plt.title("Тестовые и прогнозные значения Support Vector Machines")
plt.plot(y_pred_DT, label = "Прогноз", color = 'red')
plt.plot(y_test.values, label = "Тест", color = 'darkgreen')
plt.xlabel("Количество наблюдений")
plt.ylabel("Модуль упругости при растяжении, ГПа")
plt.legend()
plt.grid(True);


# # Гребневая регрессия (ридж-регрессия)

# In[98]:


regressor_R_2 = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
regressor_R_2.fit(x_train_2, y_train_2)


# In[99]:


y_pred_R_2 = regressor_R_2.predict(x_test_2)


# In[100]:


print('Mean Absolute Error:', mean_absolute_error(y_test_2, y_pred_R_2))
print('Mean Squared Error:', mean_squared_error(y_test_2, y_pred_R_2))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_2, y_pred_R_2)))
print ('R Squared:', r2_score(y_test_2, y_pred_R_2))


# In[101]:


R_Squared_R_2=r2_score(y_test_2, y_pred_R_2)


# In[102]:


print("Train score: {:.2f}".format(regressor_R_2.score(x_train_2, y_train_2)))
print("Test score: {:.2f}".format(regressor_R_2.score(x_test_2, y_test_2)))


# In[103]:


plt.figure(figsize = (10, 7))
plt.title("Тестовые и прогнозные значения RidgeCV")
plt.plot(y_pred_DT, label = "Прогноз", color = 'red')
plt.plot(y_test.values, label = "Тест", color = 'darkgreen')
plt.xlabel("Количество наблюдений")
plt.ylabel("Модуль упругости при растяжении, ГПа")
plt.legend()
plt.grid(True);


# # Метод градиентного бустинга - Gradient Boosting Regressor

# In[104]:


regressor_GBR_2 = GradientBoostingRegressor( random_state=0)
regressor_GBR_2.fit(x_train_2, y_train_2)


# In[105]:


y_pred_GBR_2 = regressor_GBR_2.predict(x_test_2)


# In[106]:


print('Mean Absolute Error:', mean_absolute_error(y_test_2, y_pred_GBR_2))
print('Mean Squared Error:', mean_squared_error(y_test_2, y_pred_GBR_2))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_2, y_pred_GBR_2)))
print ('R Squared:', r2_score(y_test_2, y_pred_GBR_2))


# In[107]:


R_Squared_GBR_2=r2_score(y_test_2, y_pred_GBR_2)


# In[108]:


print("Train score: {:.2f}".format(regressor_GBR_2.score(x_train_2, y_train_2)))
print("Test score: {:.2f}".format(regressor_GBR_2.score(x_test_2, y_test_2)))


# In[109]:


plt.figure(figsize = (10, 7))
plt.title("Тестовые и прогнозные значения GradientBoostingRegressor")
plt.plot(y_pred_DT, label = "Прогноз", color = 'red')
plt.plot(y_test.values, label = "Тест", color = 'darkgreen')
plt.xlabel("Количество наблюдений")
plt.ylabel("Модуль упругости при растяжении, ГПа")
plt.legend()
plt.grid(True);


# In[110]:


R_Squared_df_norm_2 = {'Регрессор': ['Support Vector', 'RandomForest',  'GradientBoosting', 'KNeighbors', 'DecisionTree','Lasso','Ridge',], 'R_Squared': [R_Squared_SVR_2,R_Squared_RF_2,R_Squared_GBR_2,R_Squared_KNN_2, R_Squared_DT_2, R_Squared_LS_2,R_Squared_R_2]}
R_Squared_df_norm_2 = pd.DataFrame(R_Squared_df_norm_2)


# In[111]:


R_Squared_df_norm_2


# В прогнозиовании- модуль упругости при растяжении наилучший результат показал алгоритм
# -GradientBoosting . Вцелом ,за некоторым исключением,алгоритмы регрессии показали нелохие
# результаты.Суммируя показатели ,алгоритм GradientBoosting показал хорошие результаты
# по метрике -R_Squared.KNeighbors показал самые противоичивые результаты.Хорошо проявив себя
# в прогнозировании-ПРОЧНОСТЬ ПРИ РАСТЯЖЕНИИ,в прогнозировании - модуль упругости при растяжении
# покозал очень плохие цифы.

# In[ ]:




