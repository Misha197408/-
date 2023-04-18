#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# Импортирование необходимых модулей и атрибутов
from sklearn import linear_model
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from matplotlib import pyplot
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from keras.wrappers.scikit_learn import KerasRegressor
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Activation, Dense
from tensorflow.keras.layers import Dropout


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


x=df[[ 'Плотность, кг/м3',
       'модуль упругости, ГПа', 'Количество отвердителя, м.%',
       'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
       'Поверхностная плотность, г/м2', 'Модуль упругости при растяжении, ГПа',
       'Прочность при растяжении, МПа', 'Потребление смолы, г/м2',
       'Угол нашивки', 'Шаг нашивки', 'Плотность нашивки']]
# целевую переменную поместим в переменную y
y=df[['Соотношение матрица-наполнитель']]


# In[6]:


x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size = 0.3, 
                                                    random_state = 42)


# # Построим простую линейную модель 1

# In[7]:


normalizer = tf.keras.layers.Normalization(axis=-1)


# In[8]:


normalizer.adapt(np.array(x))


# In[9]:


model_1 = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])
model_1.summary()


# In[10]:


model_1.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error')


# In[11]:



history = model_1.fit(
    x_train,
    y_train,
    epochs=100,
    verbose=1,
    validation_split=0.2)


# In[12]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[13]:


def plot_loss(history, lim = [0, 10]):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim(lim)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)


# In[14]:


plot_loss(history)


# In[15]:


y_pred_model_1 = model_1.predict(x_test)


# In[16]:


a = plt.axes(aspect='equal')
plt.scatter(y_test, y_pred_model_1)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[17]:


# Зададим функцию для визуализации факт/прогноз для результатов моделей
# Посмотрим на график результата работы модели
def actual_and_predicted_plot(orig, predict, var, model_name):    
    plt.figure(figsize=(17,5))
    plt.title(f'Тестовые и прогнозные значения: {model_name}')
    plt.plot(orig, label = 'Тест')
    plt.plot(predict, label = 'Прогноз')
    plt.legend(loc = 'best')
    plt.ylabel(var)
    plt.xlabel('Количество наблюдений')
    plt.show()
actual_and_predicted_plot(y_test.values, model_1.predict(x_test.values), 'Cоотношение матрица/наполнитель', 'model_1')


# #Зададим функцию для вывода метрик в виде таблицы

# In[18]:


def error (model, x_train, x_test, y_train, y_test, name = 'Model name', trg = 'Целевой параметр'):
    mae_test = mean_absolute_error(y_test, model.predict(x_test))
    mse_test = mean_squared_error(y_test, model.predict(x_test))
    mae_train = mean_absolute_error(y_train, model.predict(x_train))
    mse_train = mean_squared_error(y_train, model.predict(x_train))
    R_Squared=  r2_score(y_test, model.predict(x_test))  
    
    df_error = pd.DataFrame({
        'model':[name],
        'Target param':trg,
        'MAE(test)':mae_test,
        'MAE(train)':mae_train,
        'MSE(test)':mse_test,
        'MSE(train)':mse_train,
        'R_Squared': R_Squared
})
    return df_error


# In[19]:


df_1 = error(model_1, x_train, x_test, y_train, y_test,
       name = 'model_1', trg = 'Соотношение матрица-наполнитель')
df_1


# # Построим простую линейную модель_2 с теми же параметрами, но включим функцию callbacks

# In[20]:


# вводим функцию CALLBACKS(остановка обучения когда  целевой показатель перестает улучшаться) 
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, 
                                                     verbose=1, restore_best_weights=True)
def callbacks(pat = 10):
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=pat, 
                                                     verbose=1, restore_best_weights=True)
    return callback


# In[21]:



model_2 = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])
model_2.summary()


# In[22]:


model_2.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error')


# In[23]:


get_ipython().run_cell_magic('time', '', 'history = model_2.fit(\n    x_train,\n    y_train,\n    epochs=100,\n    verbose=1,\n    callbacks=[callback],\n    validation_split=0.2)')


# In[24]:


def plot_loss(history, lim = [0, 10]):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim(lim)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)


# In[25]:


plot_loss(history)


# In[26]:


y_pred_model_2 = model_2.predict(x_test)


# In[27]:


a = plt.axes(aspect='equal')
plt.scatter(y_test, y_pred_model_2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[28]:


def actual_and_predicted_plot(orig, predict, var, model_name):    
    plt.figure(figsize=(17,5))
    plt.title(f'Тестовые и прогнозные значения: {model_name}')
    plt.plot(orig, label = 'Тест')
    plt.plot(predict, label = 'Прогноз')
    plt.legend(loc = 'best')
    plt.ylabel(var)
    plt.xlabel('Количество наблюдений')
    plt.show()
actual_and_predicted_plot(y_test.values, model_2.predict(x_test.values), 'Cоотношение матрица/наполнитель', 'model_2')


# In[29]:


df_2 = error(model_2, x_train, x_test, y_train, y_test,
       name = 'model_2', trg = 'Соотношение матрица-наполнитель')
df_2


# # Построим простую линейную модель с теми же параметрами, но изменим оптимизатор на SGD

# In[30]:


model_3 = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])
model_3.summary()


# In[31]:


model_3.compile(
    optimizer=tf.optimizers.SGD(learning_rate=0.1),
    loss='mean_squared_error')


# In[32]:


get_ipython().run_cell_magic('time', '', 'history = model_3.fit(\n    x_train,\n    y_train,\n    epochs=100,\n    verbose=1,\n    callbacks=[callback],\n    validation_split=0.2)')


# In[33]:


def plot_loss(history, lim = [0, 10]):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim(lim)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)


# In[34]:


plot_loss(history)


# In[35]:


y_pred_model_3 = model_3.predict(x_test)


# In[36]:


a = plt.axes(aspect='equal')
plt.scatter(y_test, y_pred_model_3)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[37]:


def actual_and_predicted_plot(orig, predict, var, model_name):    
    plt.figure(figsize=(17,5))
    plt.title(f'Тестовые и прогнозные значения: {model_name}')
    plt.plot(orig, label = 'Тест')
    plt.plot(predict, label = 'Прогноз')
    plt.legend(loc = 'best')
    plt.ylabel(var)
    plt.xlabel('Количество наблюдений')
    plt.show()
actual_and_predicted_plot(y_test.values, model_3.predict(x_test.values), 'Cоотношение матрица/наполнитель', 'model_3')


# In[38]:


df_3 = error(model_3, x_train, x_test, y_train, y_test,
       name = 'model_3', trg = 'Соотношение матрица-наполнитель')
df_3


# # Построим простую линейную модель с теми же параметрами, но изменим оптимизатор на RMSprop

# In[39]:


model_4 = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])
model_4.summary()


# In[40]:


model_4.compile(
    optimizer=tf.optimizers.RMSprop(learning_rate=0.01),
    loss='mean_squared_error')


# In[41]:


get_ipython().run_cell_magic('time', '', 'history = model_4.fit(\n    x_train,\n    y_train,\n    epochs=100,\n    verbose=1,\n    callbacks=[callback],\n    validation_split=0.2)')


# In[42]:


def plot_loss(history, lim = [0, 10]):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim(lim)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)


# In[43]:


plot_loss(history)


# In[44]:


y_pred_model_4 = model_4.predict(x_test)


# In[45]:


a = plt.axes(aspect='equal')
plt.scatter(y_test, y_pred_model_4)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[46]:


def actual_and_predicted_plot(orig, predict, var, model_name):    
    plt.figure(figsize=(17,5))
    plt.title(f'Тестовые и прогнозные значения: {model_name}')
    plt.plot(orig, label = 'Тест')
    plt.plot(predict, label = 'Прогноз')
    plt.legend(loc = 'best')
    plt.ylabel(var)
    plt.xlabel('Количество наблюдений')
    plt.show()
actual_and_predicted_plot(y_test.values, model_4.predict(x_test.values), 'Cоотношение матрица/наполнитель', 'model_4')


# In[47]:


df_4 = error(model_4, x_train, x_test, y_train, y_test,
       name = 'model_4', trg = 'Соотношение матрица-наполнитель')
df_4


# In[48]:


df_result = pd.concat([df_1, df_2, df_3, df_4], axis=0).reset_index(drop = True)
df_result


# # Построение многослойного персетрона

# In[49]:


def build_and_compile_model(normalizer):
    model = keras.Sequential([
      normalizer,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss='mean_squared_error')
    return model


# In[50]:


mlp_1 = build_and_compile_model(normalizer)
mlp_1.summary()


# In[51]:


get_ipython().run_cell_magic('time', '', 'history = mlp_1.fit(\n    x_train,\n    y_train,\n    epochs=1023,\n    verbose=1,\n    callbacks=[callback],\n    validation_split=0.2)')


# In[52]:


def plot_loss(history, lim = [0, 10]):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim(lim)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)


# In[53]:


plot_loss(history)


# In[54]:


y_pred_mlp_1 = mlp_1.predict(x_test)


# In[55]:


a = plt.axes(aspect='equal')
plt.scatter(y_test, y_pred_mlp_1)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[56]:


def actual_and_predicted_plot(orig, predict, var, model_name):    
    plt.figure(figsize=(17,5))
    plt.title(f'Тестовые и прогнозные значения: {model_name}')
    plt.plot(orig, label = 'Тест')
    plt.plot(predict, label = 'Прогноз')
    plt.legend(loc = 'best')
    plt.ylabel(var)
    plt.xlabel('Количество наблюдений')
    plt.show()
actual_and_predicted_plot(y_test.values, mlp_1.predict(x_test.values), 'Cоотношение матрица/наполнитель', 'mlp_1')


# In[57]:


df_mlp_1 = error(mlp_1, x_train, x_test, y_train, y_test,
    name = 'mlp_1', trg = 'Соотношение матрица-наполнитель')
df_mlp_1


# # Увеличим число слоев и изменим число нейронов

# In[58]:


def build_and_compile_model(normalizer):
    model_2 = keras.Sequential([
      normalizer,
      layers.Dense(32, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(128, activation='relu'),  
      layers.Dense(1)
    ])

    model_2.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss='mean_squared_error')
    return model_2


# In[59]:


mlp_2 = build_and_compile_model(normalizer)
mlp_2.summary()


# In[60]:


get_ipython().run_cell_magic('time', '', 'history = mlp_2.fit(\n    x_train,\n    y_train,\n    epochs=1000,\n    verbose=1,\n    callbacks=[callback],\n    validation_split=0.2)')


# In[61]:


def plot_loss(history, lim = [0, 10]):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim(lim)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)


# In[62]:


plot_loss(history)


# In[63]:


y_pred_mlp_2 = mlp_2.predict(x_test)


# In[64]:


a = plt.axes(aspect='equal')
plt.scatter(y_test, y_pred_mlp_2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[65]:


def actual_and_predicted_plot(orig, predict, var, model_name):    
    plt.figure(figsize=(17,5))
    plt.title(f'Тестовые и прогнозные значения: {model_name}')
    plt.plot(orig, label = 'Тест')
    plt.plot(predict, label = 'Прогноз')
    plt.legend(loc = 'best')
    plt.ylabel(var)
    plt.xlabel('Количество наблюдений')
    plt.show()
actual_and_predicted_plot(y_test.values, mlp_2.predict(x_test.values), 'Cоотношение матрица/наполнитель', 'mlp_2')


# In[66]:


df_mlp_2 = error(mlp_2, x_train, x_test, y_train, y_test,
    name = 'mlp_2', trg = 'Соотношение матрица-наполнитель')
df_mlp_2


# # число слоев оставим , изменим число нейронов

# In[67]:


def build_and_compile_model(normalizer):
    model_3 = keras.Sequential([
      normalizer,
      layers.Dense(64, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dense(256, activation='relu'),  
      layers.Dense(1)
    ])

    model_3.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss='mean_squared_error')
    return model_3


# In[68]:


mlp_3 = build_and_compile_model(normalizer)
mlp_3.summary()


# In[69]:


get_ipython().run_cell_magic('time', '', 'history = mlp_3.fit(\n    x_train,\n    y_train,\n    epochs=1000,\n    verbose=1,\n    callbacks=[callback],\n    validation_split=0.2)')


# In[70]:


def plot_loss(history, lim = [0, 10]):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim(lim)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)


# In[71]:


plot_loss(history)


# In[72]:


y_pred_mlp_3 = mlp_3.predict(x_test)


# In[73]:


a = plt.axes(aspect='equal')
plt.scatter(y_test, y_pred_mlp_3)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[74]:


def actual_and_predicted_plot(orig, predict, var, model_name):    
    plt.figure(figsize=(17,5))
    plt.title(f'Тестовые и прогнозные значения: {model_name}')
    plt.plot(orig, label = 'Тест')
    plt.plot(predict, label = 'Прогноз')
    plt.legend(loc = 'best')
    plt.ylabel(var)
    plt.xlabel('Количество наблюдений')
    plt.show()
actual_and_predicted_plot(y_test.values, mlp_3.predict(x_test.values), 'Cоотношение матрица/наполнитель', 'mlp_3')


# In[75]:


df_mlp_3 = error(mlp_3, x_train, x_test, y_train, y_test,
    name = 'mlp_3', trg = 'Соотношение матрица-наполнитель')
df_mlp_3


# # Построим многослойную модель с теми же параметрами, но изменим оптимизатор на SGD

# In[76]:


def build_and_compile_model(normalizer):
    model_4 = keras.Sequential([
      normalizer,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
    ])

    model_4.compile(optimizer=tf.keras.optimizers.SGD(0.001),
                loss='mean_squared_error')
    return model_4


# In[77]:


mlp_4 = build_and_compile_model(normalizer)
mlp_4.summary()


# In[78]:


get_ipython().run_cell_magic('time', '', 'history = mlp_4.fit(\n    x_train,\n    y_train,\n    epochs=1000,\n    verbose=1,\n    callbacks=[callback],\n    validation_split=0.2)')


# In[79]:


def plot_loss(history, lim = [0, 10]):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim(lim)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
plot_loss(history)   


# In[80]:


y_pred_mlp_4 = mlp_4.predict(x_test)


# In[81]:


# Зададим функцию для визуализации факт/прогноз для результатов моделей
# Посмотрим на график результата работы модели
def actual_and_predicted_plot(orig, predict, var, model_name):    
    plt.figure(figsize=(17,5))
    plt.title(f'Тестовые и прогнозные значения: {model_name}')
    plt.plot(orig, label = 'Тест')
    plt.plot(predict, label = 'Прогноз')
    plt.legend(loc = 'best')
    plt.ylabel(var)
    plt.xlabel('Количество наблюдений')
    plt.show()
actual_and_predicted_plot(y_test.values, mlp_4.predict(x_test.values), 'Cоотношение матрица/наполнитель', 'mlp_4')


# In[82]:


a = plt.axes(aspect='equal')
plt.scatter(y_test,y_pred_mlp_4)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[83]:


df_mlp_4 = error(mlp_4, x_train, x_test, y_train, y_test,
    name = 'mlp_4', trg = 'Соотношение матрица-наполнитель')
df_mlp_4


# In[84]:


def build_and_compile_model(normalizer):
    model_5 = keras.Sequential([
      normalizer,
      layers.Dense(32, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(128, activation='relu'),  
      layers.Dense(1)
    ])

    model_5.compile(optimizer=tf.keras.optimizers.SGD(0.001),
                loss='mean_squared_error')
    return model_5


# In[85]:


mlp_5 = build_and_compile_model(normalizer)
mlp_5.summary()


# In[86]:


get_ipython().run_cell_magic('time', '', 'history = mlp_5.fit(\n    x_train,\n    y_train,\n    epochs=1000,\n    verbose=1,\n    callbacks=[callback],\n    validation_split=0.2)')


# In[87]:


def plot_loss(history, lim = [0, 10]):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim(lim)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
plot_loss(history)


# In[88]:


y_pred_mlp_5 = mlp_5.predict(x_test)


# In[89]:


a = plt.axes(aspect='equal')
plt.scatter(y_test,y_pred_mlp_5 )
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[90]:


def actual_and_predicted_plot(orig, predict, var, model_name):    
    plt.figure(figsize=(17,5))
    plt.title(f'Тестовые и прогнозные значения: {model_name}')
    plt.plot(orig, label = 'Тест')
    plt.plot(predict, label = 'Прогноз')
    plt.legend(loc = 'best')
    plt.ylabel(var)
    plt.xlabel('Количество наблюдений')
    plt.show()
actual_and_predicted_plot(y_test.values, mlp_5.predict(x_test.values), 'Cоотношение матрица/наполнитель', 'mlp_5')


# In[91]:


df_mlp_5 = error(mlp_5, x_train, x_test, y_train, y_test,
    name = 'mlp_5', trg = 'Соотношение матрица-наполнитель')
df_mlp_5


# In[92]:


def build_and_compile_model(normalizer):
    model_6 = keras.Sequential([
      normalizer,
      layers.Dense(64, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dense(256, activation='relu'),  
      layers.Dense(1)
    ])

    model_6.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss='mean_squared_error')
    return model_6


# In[93]:


mlp_6 = build_and_compile_model(normalizer)
mlp_6.summary()


# In[94]:


get_ipython().run_cell_magic('time', '', 'history = mlp_6.fit(\n    x_train,\n    y_train,\n    epochs=1000,\n    verbose=1,\n    callbacks=[callback],\n    validation_split=0.2)')


# In[95]:


def plot_loss(history, lim = [0, 10]):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim(lim)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
plot_loss(history)


# In[96]:


y_pred_mlp_6 = mlp_6.predict(x_test)


# In[97]:


a = plt.axes(aspect='equal')
plt.scatter(y_test,y_pred_mlp_6  )
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[98]:


def actual_and_predicted_plot(orig, predict, var, model_name):    
    plt.figure(figsize=(17,5))
    plt.title(f'Тестовые и прогнозные значения: {model_name}')
    plt.plot(orig, label = 'Тест')
    plt.plot(predict, label = 'Прогноз')
    plt.legend(loc = 'best')
    plt.ylabel(var)
    plt.xlabel('Количество наблюдений')
    plt.show()
actual_and_predicted_plot(y_test.values, mlp_6.predict(x_test.values), 'Cоотношение матрица/наполнитель', 'mlp_6')


# In[99]:


df_mlp_6 = error(mlp_6, x_train, x_test, y_train, y_test,
    name = 'mlp_6', trg = 'Соотношение матрица-наполнитель')
df_mlp_6


# In[100]:


df_mlp_result = pd.concat([df_mlp_1, df_mlp_2, df_mlp_3, df_mlp_4, df_mlp_5, df_mlp_6, ], axis=0).reset_index(drop = True)
df_mlp_result


# # Добавим слой дропаут

# In[101]:


def build_and_compile_model(normalizer):
    model_7 = keras.Sequential([
      normalizer,
      layers.Dense(64, activation='relu'),
      layers.Dropout(0.1),
      layers.Dense(128, activation='relu'),
      layers.Dropout(0.1),  
      layers.Dense(256, activation='relu'),
      layers.Dropout(0.1),  
      layers.Dense(1)
    ])

    model_7.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss='mean_squared_error')
    return model_7


# In[102]:


mlp_7 = build_and_compile_model(normalizer)
mlp_7.summary()


# In[103]:


get_ipython().run_cell_magic('time', '', 'history = mlp_7.fit(\n    x_train,\n    y_train,\n    epochs=1000,\n    verbose=1,\n    callbacks=[callback],\n    validation_split=0.2)')


# In[104]:


def plot_loss(history, lim = [0, 10]):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim(lim)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
plot_loss(history)


# In[105]:


y_pred_mlp_7 = mlp_7.predict(x_test)


# In[106]:


a = plt.axes(aspect='equal')
plt.scatter(y_test, y_pred_mlp_7)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[107]:


def actual_and_predicted_plot(orig, predict, var, model_name):    
    plt.figure(figsize=(17,5))
    plt.title(f'Тестовые и прогнозные значения: {model_name}')
    plt.plot(orig, label = 'Тест')
    plt.plot(predict, label = 'Прогноз')
    plt.legend(loc = 'best')
    plt.ylabel(var)
    plt.xlabel('Количество наблюдений')
    plt.show()
actual_and_predicted_plot(y_test.values, mlp_7.predict(x_test.values), 'Cоотношение матрица/наполнитель', 'mlp_7')


# In[108]:


df_mlp_7 = error(mlp_7, x_train, x_test, y_train, y_test,
    name = 'mlp_7', trg = 'Соотношение матрица-наполнитель')
df_mlp_7


# In[109]:


df_END = pd.concat([df_result, df_mlp_result, df_mlp_7], axis=0).reset_index(drop = True)
df_END


# # Метод GridSearchCV для нейросети

# In[110]:


def create_model_GSCV(lyrs=[32], act='softmax', optimizer='adam', dr=0.1):
    
    seed = 7
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    model_GSCV = Sequential()
    model_GSCV.add(Dense(lyrs[0], input_dim=x_train.shape[1], activation=act)) 
    for i in range(1,len(lyrs)):
        model_GSCV.add(Dense(lyrs[i], activation=act))
    
    model_GSCV.add(Dropout(dr))
    model_GSCV.add(Dense(1))  # выходной слой
    
    model_GSCV.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
 
    return model_GSCV


# # Ищем оптимальное число epochs и размер batch_size

# In[111]:


model_GSCV= KerasRegressor(build_fn=create_model_GSCV, verbose=0)

batch_size = [10, 20, 30, 40, 50]
epochs = [10, 50, 100]

param_grid = dict(batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator=model_GSCV, param_grid=param_grid, n_jobs=1,cv=10)
grid_result = grid.fit(x_train, y_train)

#summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# # Ищем optimizer

# In[112]:


model_GSCV = KerasRegressor(build_fn=create_model_GSCV, epochs=100, batch_size=10, verbose=0)

optimizer = ['SGD', 'RMSprop',  'Adam', ]
param_grid = dict(optimizer=optimizer)

grid = GridSearchCV(estimator=model_GSCV, param_grid=param_grid, cv=10, verbose=2)
grid_result = grid.fit(x_train, y_train)


# In[113]:


# результаты
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# # Ищем колличество слоев

# In[114]:


model_GSCV = KerasRegressor(build_fn=create_model_GSCV, epochs=100, batch_size=10, verbose=0)

layers = [[64, 64], [32, 64, 128], [64, 128, 256]]
param_grid = dict(lyrs=layers)

grid = GridSearchCV(estimator=model_GSCV, param_grid=param_grid, cv=10, verbose=2)
grid_result = grid.fit(x_train, y_train)


# In[115]:


# результаты
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# # Ищем оптимальные параметры для слоев Dropout

# In[116]:


model_GSCV =  KerasRegressor(build_fn=create_model_GSCV, epochs=100, batch_size=10, verbose=0)

drops = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
param_grid = dict(dr=drops)

grid = GridSearchCV(estimator=model_GSCV, param_grid=param_grid, cv=10, verbose=2)
grid_result = grid.fit(x_train, y_train)


# In[117]:


# результаты
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[134]:


from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Activation, Dense
from tensorflow.keras.layers import Dropout


# In[ ]:





# In[ ]:





# In[ ]:





# In[129]:


def build_and_compile_model(normalizer):
    model_8 = keras.Sequential([
      normalizer,
      layers.Dense(64, activation='relu'),
      layers.Dropout(0.1),
      layers.Dense(128, activation='relu'),
      layers.Dropout(0.1),  
      layers.Dense(256, activation='relu'),
      layers.Dropout(0.1),  
      layers.Dense(1)
    ])

    model_8.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss='mean_squared_error')
    return model_8


# In[130]:


mlp_8 = build_and_compile_model(normalizer)
mlp_8.summary()


# In[131]:


get_ipython().run_cell_magic('time', '', 'history = mlp_8.fit(\n    x_train,\n    y_train,\n    epochs=1000,\n    verbose=1,\n    callbacks=[callback],\n    validation_split=0.2)')


# In[132]:


def plot_loss(history, lim = [0, 10]):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim(lim)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
plot_loss(history)


# In[136]:


mlp_8.evaluate(x_test, y_test)


# In[138]:


mlp_8.save('App/mlp_8/NEIRO_1')


# In[140]:


mlp_8_loaded = keras.models.load_model('App/mlp_8/NEIRO_1')


# In[141]:


mlp_8_loaded.evaluate(x_test, y_test)


# In[142]:


print(tf.__version__)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




