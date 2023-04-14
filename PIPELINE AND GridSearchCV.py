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
from sklearn.model_selection import train_test_split, GridSearchCV, KFold,RepeatedKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn import set_config
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_excel(r"C:\Users\55944\Desktop\888\data_iqr_2.xlsx")
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.head(5)


# In[3]:


norm_scaler=Normalizer()
df_norm=norm_scaler.fit_transform(np.array(df[['Соотношение матрица-наполнитель','Плотность, кг/м3','модуль упругости, ГПа','Количество отвердителя, м.%','Содержание эпоксидных групп,%_2','Температура вспышки, С_2','Поверхностная плотность, г/м2','Модуль упругости при растяжении, ГПа','Прочность при растяжении, МПа','Потребление смолы, г/м2','Угол нашивки','Шаг нашивки','Плотность нашивки']]))
df_norm[:1]


# In[4]:


#возвращаем названия столбцов,нормализованную матрицу
df_norm=pd.DataFrame(data=df_norm,columns=['Соотношение матрица-наполнитель','Плотность, кг/м3','модуль упругости, ГПа','Количество отвердителя, м.%','Содержание эпоксидных групп,%_2','Температура вспышки, С_2','Поверхностная плотность, г/м2','Модуль упругости при растяжении, ГПа','Прочность при растяжении, МПа','Потребление смолы','Угол нашивки','Шаг нашивки','Плотность нашивки'])
df_norm


# In[5]:


df_norm.columns


# In[ ]:





# In[6]:


x=df_norm[['Соотношение матрица-наполнитель', 'Плотность, кг/м3',
       'модуль упругости, ГПа', 'Количество отвердителя, м.%',
       'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
       'Поверхностная плотность, г/м2', 'Модуль упругости при растяжении, ГПа',
       'Потребление смолы', 'Угол нашивки',
       'Шаг нашивки', 'Плотность нашивки']]
# целевую переменную поместим в переменную y
y=df_norm[['Прочность при растяжении, МПа']]


# In[7]:


x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size = 0.3, 
                                                    random_state = 42)


# # PIPELINE алгоритма регрессии

# # kNN

# In[8]:


set_config(display="diagram")


# In[9]:


pipe = make_pipeline( KNeighborsRegressor())
pipe.fit(x_train, y_train)


# In[10]:


y_pred_KNN = pipe.predict(x_test)


# In[11]:


print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_KNN))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_KNN))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_KNN)))
print ('R_Squared:', r2_score(y_test, y_pred_KNN))


# # Регрессия дерева решений

# In[12]:


pipe = make_pipeline(DecisionTreeRegressor())
pipe.fit(x_train, y_train)


# In[13]:


y_pred_DT = pipe.predict(x_test)


# In[14]:


print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_DT))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_DT))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_DT)))
print ('R_Squared:', r2_score(y_test, y_pred_DT))


# # Регрессия случайного леса

# In[15]:


pipe = make_pipeline(RandomForestRegressor())
pipe.fit(x_train, y_train)


# In[16]:


y_pred_RF = pipe.predict(x_test)


# In[17]:


print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_RF))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_RF))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_RF)))
print ('R_Squared:', r2_score(y_test, y_pred_RF))


# In[18]:


pipe = make_pipeline(GradientBoostingRegressor())
pipe.fit(x_train, y_train)


# In[19]:


y_pred_GBR = pipe.predict(x_test)


# In[20]:


print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_GBR))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_GBR))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_GBR)))
print ('R_Squared:', r2_score(y_test, y_pred_GBR))


# # Поиск по сетке с использованием конвейера 

# # kNN

# In[21]:


pipe = make_pipeline( KNeighborsRegressor())


# Определяем параметры

# In[22]:


param_grid = {
    "kneighborsregressor__n_neighbors": [3, 5, 8, 12, 15],
    "kneighborsregressor__weights": ["uniform", "distance"],
}


# In[23]:


grid = GridSearchCV(pipe, param_grid=param_grid, scoring="neg_mean_squared_error", cv=5)


# In[24]:


grid.fit(x_train, y_train)


# In[27]:


y_pred_knn = grid.predict(x_test)


# In[30]:


print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_knn))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_knn))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_knn)))
print ('R_Squared:', r2_score(y_test, y_pred_knn))


# In[31]:


print(grid.best_estimator_)


# # DecisionTreeRegressor

# In[68]:


pipe = make_pipeline(DecisionTreeRegressor())


# In[69]:


criterion = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
splitter = ['best', 'random']
max_depth = [3,5,7,9,11]
min_samples_leaf = [100,150,200]
min_samples_split = [200,250,300]
max_features = ['auto', 'sqrt', 'log2']


# In[70]:


param_grid = {'decisiontreeregressor__criterion': criterion,
              'decisiontreeregressor__splitter': splitter,
              'decisiontreeregressor__max_depth': max_depth,
              'decisiontreeregressor__min_samples_split': min_samples_split,
              'decisiontreeregressor__min_samples_leaf': min_samples_leaf,
              'decisiontreeregressor__max_features': max_features}


# In[71]:


grid_DTR = GridSearchCV(pipe, param_grid=param_grid, scoring="neg_mean_squared_error", cv=5)


# In[72]:


grid_DTR.fit(x_train, y_train)


# In[74]:


y_pred_DTR = grid.predict(x_test)


# In[75]:


print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_DTR))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_DTR))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_DTR)))
print ('R_Squared:', r2_score(y_test, y_pred_DTR))


# # pipeline для выбора модели

# In[146]:


pipe = Pipeline(
    [
        
        ("regressor", RandomForestRegressor()),
    ]
)


# In[147]:


param_grid = {
    
    "regressor": [
        KNeighborsRegressor(),
        LinearRegression(),
        RandomForestRegressor(random_state=42),
        DecisionTreeRegressor(random_state=42),
        XGBRegressor(random_state=42),
    ],
}


# In[149]:


grid_pip = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
)


# In[152]:


grid_pip.fit(x_train, y_train)


# In[154]:


print(np.sqrt(-grid_pip.best_score_))

print(grid_pip.best_estimator_)


# In[155]:


result = pd.DataFrame(grid_pip.cv_results_)


# In[156]:


result


# Теперь, когда мы сузили круг алгоритмов, которые хорошо работают с этим набором данных, мы можем еще больше улучшить результат, настроив параметры этих моделей по отдельности с разными настройками. Здесь мы используем отдельные словари для каждого из алгоритмов, которые хотим настроить.

# In[157]:


pipe = Pipeline(
    [
        
        ("regressor", RandomForestRegressor()),
    ]
)


# In[158]:


param_grid = [
    {
        "regressor": [RandomForestRegressor(random_state=42)],
        "regressor__n_estimators": [100, 300, 500, 1000],
        "regressor__max_depth": [3, 5, 8, 15],
        "regressor__max_features": ["log2", "sqrt", "auto"],
    },
    {
        "regressor": [XGBRegressor(random_state=42)],
        "regressor__max_depth": [3, 5, 8, 15],
        "regressor__learning_rate": [0.1, 0.01, 0.05],
        "regressor__gamma": [0, 0.25, 1.0],
        "regressor__lambda": [0, 1.0, 10.0],
    },
]


# In[159]:


grid_pip_1 = GridSearchCV(pipe, param_grid=param_grid, scoring="neg_mean_squared_error", cv=5)
grid_pip_1.fit(x_train, y_train)


# In[161]:


print(grid_pip_1.best_estimator_)


# In[163]:


print(np.sqrt(-grid_pip_1.best_score_))


# На всех   использованых в данной работе  алгоритмах и медодах настройки и автоматизации
# были показаны достойные результаты по метрикам используемых в задачах регессии.

# In[ ]:




