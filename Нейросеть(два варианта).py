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
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


# In[2]:


data_iqr_new = pd.read_excel(r'C:\Users\55944\Desktop\888/data_iqr_new.xlsx')
data_iqr_new.drop(['Unnamed: 0'], axis=1,inplace=True)


# In[3]:


data_iqr_new


# In[4]:


# посмотрим с каким типом переменных нам предстоит работать
# для этого есть метод .info()
data_iqr_new.info()


# In[5]:


# посмотрим на основные статистические показатели (summary statistics)
# с помощью метода .describe()
data_iqr_new.describe()


# In[6]:


# проверим, есть ли пропущенные значения

data_iqr_new.isnull().sum()


# In[7]:


# посчитаем коэффициент корреляции для всего датафрейма и округлим значение
# получается корреляционная матрица
corr_matrix=data_iqr_new.corr()
corr_matrix


# In[8]:


data_iqr_new.columns


# In[9]:


# отберем признаки и поместим их в переменную X
x=data_iqr_new[['Плотность, кг/м3',
       'модуль упругости, ГПа', 'Количество отвердителя, м.%',
       'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
       'Поверхностная плотность, г/м2', 'Модуль упругости при растяжении, ГПа',
       'Прочность при растяжении, МПа', 'Потребление смолы, г/м2',
       'Угол нашивки', 'Шаг нашивки', 'Плотность нашивки']]


# In[10]:


# целевую переменную поместим в переменную y
y=data_iqr_new[['Соотношение матрица-наполнитель']]


# In[11]:


print(type(x), type(y))


# In[12]:


# разобьем данные на обучающую и тестовую выборку
# размер тестовой выборки составит 30%
# также зададим точку отсчета для воспроизводимости
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size = 0.3, 
                                                    random_state = 42)


# In[13]:


# посмотрим на новую размерность обучающей
print(x_train.shape, y_train.shape)

# и тестовой выборки
print(x_test.shape, y_test.shape)


# In[14]:


normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(x_train))
def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(0.001),metrics=['mae'])
  return model
model = build_and_compile_model(normalizer)
model.summary()


# In[15]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(\n    x_train,\n    y_train,\n    validation_split=0.2,\n    verbose=1, epochs=100)')


# In[16]:


mae=model.evaluate(x_test,y_test,verbose=1)


# In[17]:


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Средняя абсолютная ошибка')
  plt.legend()
  plt.grid(True)


# In[18]:


plot_loss(history)


# In[19]:


y_pred_model = model.predict(x_test)

print('Model Results:')
print('Model_MAE: ', round(mean_absolute_error(y_test, y_pred_model)))
print('Model_MAPE: {:.2f}'.format(mean_absolute_percentage_error(y_test, y_pred_model)))
print("Test score: {:.2f}".format(mean_squared_error(y_test, y_pred_model)))


# In[20]:


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
actual_and_predicted_plot(y_test.values, model.predict(x_test.values), 'Cоотношение матрица/наполнитель', 'Keras_neuronet')


# In[21]:


test_predictions = model.predict(x_test).flatten()

a = plt.axes(aspect = 'equal')
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[22]:


pip install scikeras


# In[23]:


from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[24]:


from tensorflow.keras.layers import Dropout


# In[25]:


def create_model_1(lyrs=[32], act='softmax', optimizer='adam', dr=0.1):
    
    seed = 7
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    model_1 = Sequential()
    model_1.add(Dense(lyrs[0], input_dim=x_train.shape[1], activation=act)) 
    for i in range(1,len(lyrs)):
        model_1.add(Dense(lyrs[i], activation=act))
    
    model_1.add(Dropout(dr))
    model_1.add(Dense(1))  # выходной слой
    
    model_1.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
 
    return model_1


# In[26]:


model_1 = KerasRegressor(build_fn=create_model_1, verbose=0)

batch_size = [10, 20, 30, 40, 50]
epochs = [10, 50, 100]

param_grid = dict(batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator=model_1, param_grid=param_grid, n_jobs=1,cv=10)
grid_result = grid.fit(x_train, y_train)

#summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[27]:


model_1 = KerasRegressor(build_fn=create_model_1, epochs=100, batch_size=10, verbose=0)

optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Nadam']
param_grid = dict(optimizer=optimizer)

grid = GridSearchCV(estimator=model_1, param_grid=param_grid, cv=10, verbose=2)
grid_result = grid.fit(x_train, y_train)


# In[28]:


# результаты
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[29]:


model_1 = KerasRegressor(build_fn=create_model_1, epochs=100, batch_size=10, verbose=0)

layers = [[8],[16, 2],[32, 8, 2],[12, 6, 1], [64, 64, 3], [128, 64, 16, 3]]
param_grid = dict(lyrs=layers)

grid = GridSearchCV(estimator=model_1, param_grid=param_grid, cv=10, verbose=2)
grid_result = grid.fit(x_train, y_train)


# In[30]:


# результаты
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[31]:


model_1 =  KerasRegressor(build_fn=create_model_1, epochs=100, batch_size=10, verbose=0)

drops = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
param_grid = dict(dr=drops)

grid = GridSearchCV(estimator=model_1, param_grid=param_grid, cv=10, verbose=2)
grid_result = grid.fit(x_train, y_train)


# In[32]:


# результаты
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[45]:


# построение окончательной модели
model_1 = create_model_1(lyrs=[12,8,4], dr=0.01)

print(model_1.summary())


# In[46]:


# обучаем нейросеть, 80/20 CV
model_1_hist = model_1.fit(x_train, 
    y_train, 
    epochs = 100, 
    verbose = 1, 
    validation_split = 0.2)


# In[47]:


# оценим модель
scores = model_1.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model_1.metrics_names[1], scores[1]*100))


# In[48]:


# Посмотрим на график потерь на тренировочной и тестовой выборках
def model_1_loss_plot(model_1_hist):
    plt.figure(figsize = (17,5))
    plt.plot(model_1_hist.history['loss'],
             label = 'ошибка на обучающей выборке')
    plt.plot(model_1_hist.history['val_loss'],
            label = 'ошибка на тестовой выборке')
    plt.title('График потерь модели')
    plt.ylabel('Значение ошибки')
    plt.xlabel('Эпохи')
    plt.legend(['Oшибка на обучающей выборке', 'Ошибка на тестовой выборке'], loc='best')
    plt.show()
model_1_loss_plot(model_1_hist)


# In[49]:


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
actual_and_predicted_plot(y_test.values, model_1.predict(x_test.values), 'Cоотношение матрица/наполнитель', 'Neuronet')


# In[50]:


test_predictions = model_1.predict(x_test).flatten()

a = plt.axes(aspect = 'equal')
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[1]:


#Сохраним первый вариант нейросети и проверим ее загрузку


# In[51]:


model.save('App/model/NEIRO')


# In[52]:


model_loaded = keras.models.load_model('App/model/NEIRO')


# In[53]:


model_loaded.evaluate(x_test, y_test)


# In[54]:


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Средняя абсолютная ошибка')
  plt.legend()
  plt.grid(True)


# In[55]:


plot_loss(history)


# In[ ]:




