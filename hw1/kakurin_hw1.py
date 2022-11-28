#!/usr/bin/env python
# coding: utf-8

# # Машинное обучение
# ## Домашнее задание №1: KNN + Линейные модели

# **Срок сдачи:** 30 ноября 2021, 08:30 
# 
# **Максимально баллов:** 10 
# 
# **Штраф за опоздание:** по 2 балла за 24 часа задержки. Через 5 дней домашнее задание сгорает.
# 
# При отправлении ДЗ указывайте фамилию в названии файла. Формат сдачи будет указан чуть позже.
# 
# Используйте данный Ipython Notebook при оформлении домашнего задания.

# 
# **Штрафные баллы:**
# 
# 1. Отсутствие фамилии в имени скрипта (скрипт должен называться по аналогии со stroykova_hw1.ipynb) -1 баллов
# 2. Все строчки должны быть выполнены. Нужно, чтобы output команды можно было увидеть уже в git'е. В противном случае -1 баллов
# 
# При оформлении ДЗ нужно пользоваться данным файлом в качестве шаблона. Не нужно удалять и видоизменять написанный код и текст, если явно не указана такая возможность.

# ## KNN (5 баллов)

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_openml as fetch_mldata, fetch_20newsgroups

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from warnings import filterwarnings
from scipy import sparse
from tqdm.notebook import tqdm

SEED = int(1e9+7e7+17)

filterwarnings('ignore', category=Warning, module='sklearn')
np.random.seed(SEED)


# ##### Задание 1 (1 балл)
# Реализовать KNN в классе MyKNeighborsClassifier (обязательное условие: точность не ниже sklearn реализации)
# Разберитесь самостоятельно, какая мера расстояния используется в KNeighborsClassifier дефолтно и реализуйте свой алгоритм именно с этой мерой. 
# Для подсчета расстояний можно использовать функции [отсюда](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)

# In[2]:


class Node:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.l = None
        self.r = None
        self.feat=None
        self.med=None
    
    
class MyKNeighborsClassifier(BaseEstimator):
    
    def __init__(self, n_neighbors, algorithm='brute', leaf_size=30, metric='minkowski'):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        
    def create_kd_tree(self, feat=0, ind=None):
        cur = Node(None, None)
        if(ind is None): ind = np.array(np.arange(self.X.shape[0]))
        if(len(ind)>self.leaf_size):
            for i in range(self.X.shape[1]):
                if(self.X[:,feat+i].min() != self.X[:,feat+i].max()):
                    feat += i
                    break
            cur.feat = feat
            X_ = self.X[ind, feat]
            x = (np.array(X_.todense()).flatten() if sparse.issparse(X_) else X_)
            med = np.sort(x)[X_.shape[0]//2]
            cur.med = med
            cur.l = self.create_kd_tree((feat+1)%self.X.shape[1], ind[x<=med])
            cur.r = self.create_kd_tree((feat+1)%self.X.shape[1], ind[x>med])
        elif len(ind)==0:
            return None
        else:
            cur = Node(self.X[ind].copy(), self.y[ind].copy())
        return cur
    
    def fit(self, X, y):
        if self.algorithm=='brute':
            self.X = X.copy()
            self.y = y.copy()
        elif self.algorithm=='kd_tree':
            self.X=X
            self.y=y
            self.kd_tree = self.create_kd_tree()
            del self.X, self.y
        else:
            raise NotImplementedError
            
    def _predict_brute(self, X, X_=None, y_=None):
        if(X_ is None): X_ = self.X
        if(y_ is None): y_ = self.y
        x = cdist(X, X_, metric=self.metric)
        x = np.argsort(x, axis=1)[:, :self.n_neighbors]
        x = np.array([[y_[i] for i in row] for row in x])
        ans=[]
        for row in x:
            cnt={}
            for y in row:
                cnt[y] = cnt.setdefault(y, 0) + 1
            tmp = (-1,0)
            for y in sorted(cnt.keys()):
                if(cnt[y]>tmp[1]):
                    tmp = (y, cnt[y])
            ans.append(tmp[0])
        return np.array(ans)
    
    def _predict_kd_tree(self, X):
        ans = []
        for x in X:
            if(sparse.issparse(x)): x = np.array(x.todense()).flatten()
            cur = self.kd_tree
            feat=0
            while(cur.X is None or cur.X.shape[0]>self.leaf_size):
                cur = (cur.l if x[feat]<cur.med else cur.r)
                feat = (feat+1)%X.shape[1]
            if(sparse.issparse(X)):
                ans.append(self._predict_brute(x.reshape(1,-1), cur.X.todense(), cur.y))
            else:
                ans.append(self._predict_brute(x.reshape(1,-1), cur.X, cur.y))
        return np.array(ans).flatten()
    
    def predict(self, X):
        if(self.algorithm == 'brute'):
            return self._predict_brute(X)
        elif(self.algorithm == 'kd_tree'):
            return self._predict_kd_tree(X)
        else:
            raise NotImplementedError


# **IRIS**
# 
# В библиотеке scikit-learn есть несколько датасетов из коробки. Один из них [Ирисы Фишера](https://ru.wikipedia.org/wiki/%D0%98%D1%80%D0%B8%D1%81%D1%8B_%D0%A4%D0%B8%D1%88%D0%B5%D1%80%D0%B0)

# In[3]:


iris = datasets.load_iris()


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1, stratify=iris.target)


# In[5]:


clf = KNeighborsClassifier(n_neighbors=2, algorithm='brute')
my_clf = MyKNeighborsClassifier(n_neighbors=2, algorithm='brute')


# In[6]:


clf.fit(X_train, y_train)
my_clf.fit(X_train, y_train)


# In[7]:


sklearn_pred = clf.predict(X_test)
my_clf_pred = my_clf.predict(X_test)
assert abs( accuracy_score(y_test, my_clf_pred) -  accuracy_score(y_test, sklearn_pred ) )<0.005, "Score must be simillar"


# **Задание 2 (0.5 балла)**
# 
# Давайте попробуем добиться скорости работы на fit, predict сравнимой со sklearn для iris. Допускается замедление не более чем в 2 раза. 
# Для этого используем numpy. 

# In[8]:


get_ipython().run_line_magic('timeit', 'clf.fit(X_train, y_train)')


# In[9]:


get_ipython().run_line_magic('timeit', 'my_clf.fit(X_train, y_train)')


# In[10]:


get_ipython().run_line_magic('timeit', 'clf.predict(X_test)')


# In[11]:


get_ipython().run_line_magic('timeit', 'my_clf.predict(X_test)')


# ###### Задание 3 (1 балл)
# Добавьте algorithm='kd_tree' в реализацию KNN (использовать KDTree из sklearn.neighbors). Необходимо добиться скорости работы на fit,  predict сравнимой со sklearn для iris. Допускается замедление не более чем в 2 раза. 
# Для этого используем numpy. Точность не должна уступать значению KNN из sklearn. 

# In[12]:


clf = KNeighborsClassifier(n_neighbors=2, algorithm='kd_tree')
my_clf = MyKNeighborsClassifier(n_neighbors=2, algorithm='kd_tree')


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1, stratify=iris.target)


# In[14]:


get_ipython().run_line_magic('time', 'clf.fit(X_train, y_train)')


# In[15]:


get_ipython().run_line_magic('time', 'my_clf.fit(X_train, y_train)')


# In[16]:


get_ipython().run_line_magic('time', 'clf.predict(X_test)')


# In[17]:


get_ipython().run_line_magic('time', 'my_clf.predict(X_test)')


# In[18]:


sklearn_pred = clf.predict(X_test)
my_clf_pred = my_clf.predict(X_test)
assert abs( accuracy_score(y_test, my_clf_pred) -  accuracy_score(y_test, sklearn_pred ) )<0.005, "Score must be simillar"


# **Задание 4 (2.5 балла)**
# 
# Рассмотрим новый датасет 20 newsgroups

# In[19]:


newsgroups = fetch_20newsgroups(subset='train',remove=['headers','footers', 'quotes'])


# In[20]:


data = newsgroups['data']
target = newsgroups['target']


# Преобразуйте текстовые данные из data с помощью [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html). Словарь можно ограничить по частотности.

# In[21]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing.pool import ThreadPool
vectorizer = CountVectorizer(min_df=10, max_df=0.01)
X = vectorizer.fit_transform(data)


# In[22]:


X = np.array(X.todense())
X.shape


# *Так мы получили векторное представление наших текстов. Значит можно приступать к задаче обучения модели*

# Реализуйте разбиение выборки для кросс-валидации на 3 фолдах. Разрешено использовать sklearn.cross_validation

# In[23]:


def get_crossval_splits(X, y, cv=3):
    bounds = [X.shape[0]//cv*i for i in range(cv)] + [X.shape[0]]
    splits = []
    for i in range(cv):
        i_tr = []
        for j in range(cv):
            if i!=j:
                i_tr.extend([k for k in range(bounds[j],bounds[j+1])])
        i_ts = [k for k in range(bounds[i], bounds[i+1])]
        splits.append((i_tr, i_ts))
    return splits


# Напишите метод, позволяющий найти оптимальное количество ближайших соседей(дающее максимальную точность в среднем на валидации на 3 фолдах).
# Постройте график зависимости средней точности от количества соседей. Можно рассмотреть число соседей от 1 до 10.

# In[24]:


def neighbor_search_helper(model, X, y, i_tr, i_ts, scores, mn=1, mx=10, cv=3, args=(), kwargs={}):
    X_tr, X_ts, y_tr, y_ts = X[i_tr], X[i_ts], y[i_tr], y[i_ts]
    md = model(n_neighbors=1, *args, **kwargs)
    md.fit(X_tr, y_tr)
    for n in tqdm(range(mn, mx+1)):
        md.n_neighbors = n
        scores[n].append(accuracy_score(y_ts, md.predict(X_ts)))
    
def neighbor_search(model, X, y, mn=1, mx=10, cv=3, *args, **kwargs):
    splits = get_crossval_splits(X, y, cv)
    scores = {}
    for i in range(mn, mx+1): scores[i]=[]
    with ThreadPool(cv) as p:
        p.starmap(neighbor_search_helper, [(model, X, y, i_tr, i_ts, scores, mn, mx, cv, args, kwargs) for i_tr, i_ts in splits])
            
    # for i in range(mn, mx+1): scores[i] = np.mean(scores[i])
    return (np.argmax(np.array(list(scores.values())).mean(axis=1)) + mn, scores)


# In[25]:


ind, scores = neighbor_search(MyKNeighborsClassifier, X, target, algorithm='brute')
print(f"Optimal neighbors: {ind}, with score {np.mean(scores[ind])}")
plt.plot(np.arange(1,11), np.array(list(scores.values())).mean(axis=1))


# Как изменится качество на валидации, если:
# 
# 1. Используется косинусная метрика вместо евклидовой.
# 2. К текстам применяется TfIdf векторизацию( sklearn.feature_extraction.text.TfidfVectorizer)
# 
# Сравните модели, выберите лучшую.

# In[26]:


ind, scores = neighbor_search(MyKNeighborsClassifier, X, target, algorithm='brute', metric='cosine')
print(f"Cosine metric\nOptimal neighbors: {ind}, with score {np.mean(scores[ind])}")
vectorizer = TfidfVectorizer(min_df=10, max_df=0.01)
X = vectorizer.fit_transform(data)
X = np.array(X.todense())
ind, scores = neighbor_search(MyKNeighborsClassifier, X, target, algorithm='brute')
print(f"Tfidf vectorizer\nOptimal neighbors: {ind}, with score {np.mean(scores[ind])}")


# Загрузим  теперь test  часть нашей выборки и преобразуем её аналогично с train частью. Не забудьте, что наборы слов в train и test части могут отличаться.

# In[28]:


newsgroups = fetch_20newsgroups(subset='test',remove=['headers','footers', 'quotes'])


# Оценим точность вашей лучшей модели на test части датасета. Отличается ли оно от кросс-валидации? Попробуйте сделать выводы, почему отличается качество.

# In[27]:


test_data = newsgroups['data']
test_target = newsgroups['target']
vectorizer = CountVectorizer(min_df=10, max_df=0.01)
X = vectorizer.fit_transform(data)
X = np.array(X.todense())
X_ = vectorizer.transform(test_data)
X_ = np.array(X_.todense())
md = MyKNeighborsClassifier(n_neighbors=10, algorithm='brute', metric='cosine')
md.fit(X, target)
preds = md.predict(X_)
accuracy_score(test_target, preds)


# На тесте качество стало заметно лучше, чем на кросс-валидации. Это можно объяснить тем, что тренировочная выборка увеличилась в 1.5 раза.

# # Линейные модели (5 баллов)

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12,5)


# В этом задании мы будем реализовать линейные модели. Необходимо реализовать линейную и логистическую регрессии с L2 регуляризацией
# 
# ### Теоретическое введение
# 
# 
# 
# Линейная регрессия решает задачу регрессии и оптимизирует функцию потерь MSE 
# 
# $$L(w) =  \frac{1}{N}\left[\sum_i (y_i - a_i) ^ 2 \right], $$ где $y_i$ $-$ целевая функция,  $a_i = a(x_i) =  \langle\,x_i,w\rangle ,$ $-$ предсказание алгоритма на объекте $x_i$, $w$ $-$ вектор весов (размерности $D$), $x_i$ $-$ вектор признаков (такой же размерности $D$).
# 
# Не забываем, что здесь и далее  мы считаем, что в $x_i$ есть тождественный вектор единиц, ему соответствует вес $w_0$.
# 
# 
# Логистическая регрессия является линейным классификатором, который оптимизирует так называемый функционал log loss:
# 
# $$L(w) = - \frac{1}{N}\left[\sum_i y_i \log a_i + ( 1 - y_i) \log (1 - a_i) \right],$$
# где  $y_i  \in \{0,1\}$ $-$ метка класса, $a_i$ $-$ предсказание алгоритма на объекте $x_i$. Модель пытается предсказать апостериорую вероятность объекта принадлежать к классу "1":
# $$ p(y_i = 1 | x_i) = a(x_i) =  \sigma( \langle\,x_i,w\rangle ),$$
# $w$ $-$ вектор весов (размерности $D$), $x_i$ $-$ вектор признаков (такой же размерности $D$).
# 
# Функция $\sigma(x)$ $-$ нелинейная функция, пероводящее скалярное произведение объекта на веса в число $\in (0,1)$ (мы же моделируем вероятность все-таки!)
# 
# $$\sigma(x) = \frac{1}{1 + \exp(-x)}$$
# 
# Если внимательно посмотреть на функцию потерь, то можно заметить, что в зависимости от правильного ответа алгоритм штрафуется или функцией $-\log a_i$, или функцией $-\log (1 - a_i)$.
# 
# 
# 
# Часто для решения проблем, которые так или иначе связаны с проблемой переобучения, в функционал качества добавляют слагаемое, которое называют ***регуляризацией***. Итоговый функционал для линейной регрессии тогда принимает вид:
# 
# $$L(w) =  \frac{1}{N}\left[\sum_i (y_i - a_i) ^ 2 \right] + \frac{1}{C}R(w) $$
# 
# Для логистической: 
# $$L(w) = - \frac{1}{N}\left[\sum_i y_i \log a_i + ( 1 - y_i) \log (1 - a_i) \right] +  \frac{1}{C}R(w)$$
# 
# Самое понятие регуляризации введено основателем ВМК академиком Тихоновым https://ru.wikipedia.org/wiki/Метод_регуляризации_Тихонова
# 
# Идейно методика регуляризации заключается в следующем $-$ мы рассматриваем некорректно поставленную задачу (что это такое можно найти в интернете), для того чтобы сузить набор различных вариантов (лучшие из которых будут являться переобучением ) мы вводим дополнительные ограничения на множество искомых решений. На лекции Вы уже рассмотрели два варианта регуляризации.
# 
# $L1$ регуляризация:
# $$R(w) = \sum_{j=1}^{D}|w_j|$$
# $L2$ регуляризация:
# $$R(w) =  \sum_{j=1}^{D}w_j^2$$
# 
# С их помощью мы ограничиваем модель в  возможности выбора каких угодно весов минимизирующих наш лосс, модель уже не сможет подстроиться под данные как ей угодно. 
# 
# Вам нужно добавить соотвествущую Вашему варианту $L2$ регуляризацию.
# 
# И так, мы поняли, какую функцию ошибки будем минимизировать, разобрались, как получить предсказания по объекту и обученным весам. Осталось разобраться, как получить оптимальные веса. Для этого нужно выбрать какой-то метод оптимизации.
# 
# 
# 
# Градиентный спуск является самым популярным алгоритмом обучения линейных моделей. В этом задании Вам предложат реализовать стохастический градиентный спуск или  мини-батч градиентный спуск (мини-батч на русский язык довольно сложно перевести, многие переводят это как "пакетный", но мне не кажется этот перевод удачным). Далее нам потребуется определение **эпохи**.
# Эпохой в SGD и MB-GD называется один проход по **всем** объектам в обучающей выборки.
# * В SGD градиент расчитывается по одному случайному объекту. Сам алгоритм выглядит примерно так:
#         1) Перемешать выборку
#         2) Посчитать градиент функции потерь на одном объекте (далее один объект тоже будем называть батчем)
#         3) Сделать шаг спуска
#         4) Повторять 2) и 3) пока не пройдет максимальное число эпох.
# * В Mini Batch SGD - по подвыборке объектов. Сам алгоритм выглядит примерно так::
#         1) Перемешать выборку, выбрать размер мини-батча (от 1 до размера выборки)
#         2) Почитать градиент функции потерь по мини-батчу (не забыть поделить на  число объектов в мини-батче)
#         3) Сделать шаг спуска
#         4) Повторять 2) и 3) пока не пройдет максимальное число эпох.
# * Для отладки алгоритма реализуйте возможность  вывода средней ошибки на обучении модели по объектам (мини-батчам). После шага градиентного спуска посчитайте значение ошибки на объекте (или мини-батче), а затем усредните, например, по ста шагам. Если обучение проходит корректно, то мы должны увидеть, что каждые 100 шагов функция потерь уменьшается. 
# * Правило останова - максимальное количество эпох
#     

# ## Зачем нужны батчи?
# 
# 
# Как Вы могли заметить из теоретического введения, что в случае SGD, что в случа mini-batch GD,  на каждой итерации обновление весов  происходит только по небольшой части данных (1 пример в случае SGD, batch примеров в случае mini-batch). То есть для каждой итерации нам *** не нужна вся выборка***. Мы можем просто итерироваться по выборке, беря батч нужного размера (далее 1 объект тоже будем называть батчом).
# 
# Легко заметить, что в этом случае нам не нужно загружать все данные в оперативную память, достаточно просто считать батч с диска, обновить веса, считать диска другой батч и так далее. В целях упрощения домашней работы, прямо с диска  мы считывать не будем, будем работать с обычными numpy array. 
# 
# 
# 
# 
# 
# ## Немножко про генераторы в Python
# 
# 
# 
# Идея считывания данных кусками удачно ложится на так называемые ***генераторы*** из языка Python. В данной работе Вам предлагается не только разобраться с логистической регрессией, но  и познакомиться с таким важным элементом языка.  При желании Вы можете убрать весь код, связанный с генераторами, и реализовать логистическую регрессию и без них, ***штрафоваться это никак не будет***. Главное, чтобы сама модель была реализована правильно, и все пункты были выполнены. 
# 
# Подробнее можно почитать вот тут https://anandology.com/python-practice-book/iterators.html
# 
# 
# К генератору стоит относиться просто как к функции, которая порождает не один объект, а целую последовательность объектов. Новое значение из последовательности генерируется с помощью ключевого слова ***yield***. 
# 
# Концепция крайне удобная для обучения  моделей $-$ у Вас есть некий источник данных, который Вам выдает их кусками, и Вам совершенно все равно откуда он их берет. Под ним может скрывать как массив в оперативной памяти, как файл на жестком диске, так и SQL база данных. Вы сами данные никуда не сохраняете, оперативную память экономите.
# 
# Если Вам понравилась идея с генераторами, то Вы можете реализовать свой, используя прототип batch_generator. В нем Вам нужно выдавать батчи признаков и ответов для каждой новой итерации спуска. Если не понравилась идея, то можете реализовывать SGD или mini-batch GD без генераторов.

# In[29]:


def batch_generator(X, y, shuffle=True, batch_size=1):
    """
    Гератор новых батчей для обучения
    X          - матрица объекты-признаки
    y_batch    - вектор ответов
    shuffle    - нужно ли случайно перемешивать выборку
    batch_size - размер батча ( 1 это SGD, > 1 mini-batch GD)
    Генерирует подвыборку для итерации спуска (X_batch, y_batch)
    """
    
    # X_batch = ""
    # y_batch = ""
    order = [i for i in range(X.shape[0])]
    if(shuffle): np.random.shuffle(order)
    i=0
    for i in range(0, len(order)//batch_size*batch_size, batch_size):
        indices = order[i:i+batch_size]
        X_batch = X[indices]
        y_batch = y[indices]
        yield (X_batch, y_batch)


# In[30]:


# Теперь можно сделать генератор по данным ()
# my_batch_generator = batch_generator(X, y, shuffle=True, batch_size=100)


# In[110]:


#%%pycodestyle

def sigmoid(x):
    """
    Вычисляем значение сигмоида.
    X - выход линейной модели
    """
    
    ## Your code Here
    return 1/(1+np.exp(-np.clip(x, -100, 100)))


from sklearn.base import BaseEstimator, ClassifierMixin

class MySGDClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, batch_generator, C=1, alpha=0.01, max_epoch=10, model_type='lin_reg', batch_size=1):
        """
        batch_generator -- функция генератор, которой будем создавать батчи
        C - коэф. регуляризации
        alpha - скорость спуска
        max_epoch - максимальное количество эпох
        model_type - тим модели, lin_reg или log_reg
        batch_size - размер батча для batch_generator
        """
        
        self.C = C
        self.alpha = alpha
        self.max_epoch = max_epoch
        self.batch_generator = batch_generator
        self.errors_log = {'iter' : [], 'loss' : []}  
        self.model_type = model_type
        self.batch_size=batch_size
        self.logs={}
        
    def calc_loss(self, X_batch, y_batch):
        """
        Считаем функцию потерь по батчу 
        X_batch - матрица объекты-признаки по батчу
        y_batch - вектор ответов по батчу
        Не забудте тип модели (линейная или логистическая регрессия)!
        """
        loss = 0
        self.predict(X_batch)
        preds = self.preds
        y = y_batch.reshape(-1,1)
        if(self.model_type=='lin_reg'):
            loss = ((y-preds)**2).mean()
        else:
            preds = np.clip(preds, sigmoid(-10), sigmoid(10))
            loss = -1/X_batch.shape[0]*np.sum(y*np.log(preds) + (1-y)*np.log(1-preds))
        
        if(self.C!=0):
            loss += 1/self.C*np.sum(self.weights**2)
        return loss
    
    def calc_loss_grad(self, X_batch, y_batch):
        """
        Считаем  градиент функции потерь по батчу (то что Вы вывели в задании 1)
        X_batch - матрица объекты-признаки по батчу
        y_batch - вектор ответов по батчу
        Не забудте тип модели (линейная или логистическая регрессия)!
        """
        self.predict(X_batch)
        preds = self.preds
        y = y_batch.reshape(-1,1)
        if(self.model_type=='lin_reg'):
            loss_grad = 2*((preds-y)*X_batch).mean(axis=0).reshape(-1,1)
        else:
            preds = np.clip(preds, sigmoid(-10), sigmoid(10))
            loss_grad = -(y*(1-preds)*X_batch + (1-y)*(-1)*preds*X_batch).mean(axis=0).reshape(-1,1)
        
        if(self.C!=0):
            loss_grad += 1/self.C*self.weights.reshape(-1,1)
        return loss_grad
    
    def update_weights(self, new_grad):
        """
        Обновляем вектор весов
        new_grad - градиент по батчу
        """
        new_grad = np.clip(new_grad, -np.float64(1e4), np.float64(1e4))
        self.weights -= self.alpha * new_grad
        pass
    
    def fit(self, X, y):
        '''
        Обучение модели
        X - матрица объекты-признаки
        y - вектор ответов
        '''
        self.logs['losses']=[]
        self.logs['weights']=[]
        # Нужно инициализровать случайно веса
        self.weights = np.random.randn(X.shape[1]+1, 1)*(1/np.sqrt(X.shape[1]))
        cnt=0
        for n in range(0, self.max_epoch):
            # print(n)
            new_epoch_generator = self.batch_generator(X, y, shuffle=True, batch_size=self.batch_size)
            for batch_num, new_batch in enumerate(new_epoch_generator):
                X_batch = np.hstack([new_batch[0], np.ones((new_batch[0].shape[0], 1))])
                y_batch = new_batch[1]
                batch_grad = self.calc_loss_grad(X_batch, y_batch)
                self.update_weights(batch_grad)
                # Подумайте в каком месте стоит посчитать ошибку для отладки модели
                # До градиентного шага или после
                batch_loss = self.calc_loss(X_batch, y_batch)
                self.logs['losses'].append(batch_loss)
                self.logs['weights'].append(np.abs(self.weights).mean())
                self.errors_log['iter'].append(batch_num)
                self.errors_log['loss'].append(batch_loss)
                cnt+=1
               
        return self
        
    def predict(self, X):
        '''
        Предсказание класса
        X - матрица объекты-признаки
        Не забудте тип модели (линейная или логистическая регрессия)!
        '''
        
        # Желательно здесь использовать матричные операции между X и весами, например, numpy.dot 
        y_hat = X@self.weights
        if(self.model_type=='log_reg'): y_hat = sigmoid(y_hat)
        self.preds = y_hat.copy().reshape(-1,1)
        y_hat = ((y_hat>0.5) if(self.model_type=='log_reg') else (y_hat>0))
        return y_hat


# Запустите обе регрессии на синтетических данных. 
# 
# 
# Выведите полученные веса и нарисуйте разделяющую границу между классами (используйте только первых два веса для первых двух признаков X[:,0], X[:,1] для отображения в 2d пространство ).  

# In[111]:


def plot_decision_boundary(clf, bounds=None):
    if bounds is not None:
        plt.xlim((bounds[0], bounds[1]))
        plt.ylim((bounds[2], bounds[3]))
    x = np.arange(-6, 8, 0.1)
    y = [(-x_*clf.weights[0] - clf.weights[-1])/clf.weights[1] for x_ in x]
    plt.plot(x,y)


# In[112]:


np.random.seed(0)

C1 = np.array([[0., -0.8], [1.5, 0.8]])
C2 = np.array([[1., -0.7], [2., 0.7]])
gauss1 = np.dot(np.random.randn(200, 2) + np.array([5, 3]), C1)
gauss2 = np.dot(np.random.randn(200, 2) + np.array([1.5, 0]), C2)

X = np.vstack([gauss1, gauss2])
y = np.r_[np.ones(200), np.zeros(200)]

model = MySGDClassifier(batch_generator, max_epoch=10, model_type='log_reg', C=0, batch_size=1)
model.fit(X,y)
plot_decision_boundary(model, (X[:,0].min(), X[:,0].max(), X[:,1].min(), X[:,1].max()))

plt.scatter(X[:,0], X[:,1], c=y)


# Далее будем анализировать Ваш алгоритм. 
# Для этих заданий используйте датасет ниже.

# In[113]:


from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100000, n_features=10, 
                           n_informative=4, n_redundant=0, 
                           random_state=123, class_sep=1.0,
                           n_clusters_per_class=1)


# Покажите сходимости обеих регрессией на этом датасете: изобразите график  функции потерь, усредненной по $N$ шагам градиентого спуска, для разных `alpha` (размеров шага). Разные `alpha` расположите на одном графике. 
# 
# $N$ можно брать 10, 50, 100 и т.д. 

# In[114]:


for tp in ['lin_reg', 'log_reg']:
    for C in [1]:
        for alpha in np.arange(1e-2, 1e-1+1e-2, 3e-2):
            model = MySGDClassifier(batch_generator, max_epoch=4, model_type=tp, C=C, alpha=alpha, batch_size=100)
            model.fit(X,y)
            N=100
            plt.plot([np.mean(model.logs['losses'][i:i+N]) for i in range(len(model.logs['losses'])-N)], label=f"alpha={alpha}");
    plt.legend()
    plt.title(f"model: {tp}")
    plt.show()


# Что Вы можете сказать про сходимость метода при различных `alpha`? Какое значение стоит выбирать для лучшей сходимости?
# 
# Изобразите график среднего значения весов для обеих регрессий в зависимости от коеф. регуляризации С из `np.logspace(3, -3, 10)` 

# Для лучшей сходимости возьмём $alpha=0.1$

# In[115]:


for tp in ['lin_reg', 'log_reg']:
    dots=[[],[]]
    for C in np.logspace(3, -3, 10):
        for alpha in [1e-1]:
            model = MySGDClassifier(batch_generator, max_epoch=10, model_type=tp, C=C, alpha=alpha, batch_size=100)
            model.fit(X,y)
            dots[0].append(str(C)[:5])
            dots[1].append(model.logs['weights'][-1])
    # plt.legend()
    plt.plot(dots[1])
    plt.title(f"model: {tp}")
    plt.xticks(range(len(dots[0])), dots[0], rotation=45)
    plt.xlabel('C')
    plt.ylabel('Mean absolute weights')
    plt.show()


# Довольны ли Вы, насколько сильно уменьшились Ваши веса? 

# При малых коэффициентах регуляризации выражение $\dfrac{1}{C}R(w)$ становится очень большим по модулю, из-за чего веса начинают расходиться. Рассмотрим тот же график, но только для значений $C = 1000 .. 0.1$

# In[117]:


for tp in ['lin_reg', 'log_reg']:
    dots=[[],[]]
    for C in np.logspace(3, -1, 10):
        for alpha in [1e-1]:
            model = MySGDClassifier(batch_generator, max_epoch=10, model_type=tp, C=C, alpha=alpha, batch_size=100)
            model.fit(X,y)
            dots[0].append(str(C)[:5])
            dots[1].append(model.logs['weights'][-1])
    plt.plot(dots[1])
    plt.title(f"model: {tp}")
    plt.xticks(range(len(dots[0])), dots[0], rotation=45)
    plt.xlabel('C')
    plt.ylabel('Mean absolute weights')
    plt.show()


# При достаточно больших значениях $C$ веса модели уменьшаются при уменьшении $C$, как и ожидалось
