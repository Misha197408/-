#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error


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


norm_scaler=Normalizer()
df_norm=norm_scaler.fit_transform(np.array(df[['Соотношение матрица-наполнитель','Плотность, кг/м3','модуль упругости, ГПа','Количество отвердителя, м.%','Содержание эпоксидных групп,%_2','Температура вспышки, С_2','Поверхностная плотность, г/м2','Модуль упругости при растяжении, ГПа','Прочность при растяжении, МПа','Потребление смолы, г/м2','Угол нашивки','Шаг нашивки','Плотность нашивки']]))
df_norm[:1]


# In[6]:


df_norm=pd.DataFrame(data=df_norm,columns=['Соотношение матрица-наполнитель','Плотность, кг/м3','модуль упругости, ГПа','Количество отвердителя, м.%','Содержание эпоксидных групп,%_2','Температура вспышки, С_2','Поверхностная плотность, г/м2','Модуль упругости при растяжении, ГПа','Прочность при растяжении, МПа','Потребление смолы','Угол нашивки','Шаг нашивки','Плотность нашивки'])
df_norm


# In[7]:


# отберем признаки и поместим их в переменную X
x=df_norm[['Соотношение матрица-наполнитель', 'Плотность, кг/м3',
       'модуль упругости, ГПа', 'Количество отвердителя, м.%',
       'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
       'Поверхностная плотность, г/м2', 'Модуль упругости при растяжении, ГПа',
        'Потребление смолы', 'Угол нашивки',
       'Шаг нашивки', 'Плотность нашивки']]
# целевую переменную поместим в переменную y
y=df_norm[['Прочность при растяжении, МПа']]


# In[8]:


# разобьем данные на обучающую и тестовую выборку
# размер тестовой выборки составит 30%
# также зададим точку отсчета для воспроизводимости
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size = 0.3, 
                                                    random_state = 42)


# In[ ]:





# In[9]:


regressor_RF = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor_RF.fit(x_train, y_train) 


# In[10]:


y_pred_RF = regressor_RF.predict(x_test)


# In[11]:


print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_RF))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_RF))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_RF)))
print ('R_Squared:', r2_score(y_test, y_pred_RF))


# In[12]:


pickle.dump(regressor_RF, open('vkr.pkl', 'wb'))


# In[13]:


pip freeze


# In[14]:


print (pd.__version__)


# In[15]:


import sklearn
print(sklearn.__version__)


# In[ ]:




