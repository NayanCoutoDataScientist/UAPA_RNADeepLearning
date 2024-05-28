# Minha Primeira Rede Neural
import numpy as np
import matplotlib.pyplot as plt
#%%
# Criando a Base de Dados
np.random.seed(42)
ages = np.random.randint(low=15, high=70, size=40)
ages
#%%
label = []
for age in ages:
    if age < 30:
        label.append(0)
    else:
        label.append(1)

# random swap
for i in range (0,3):
    r = np.random.randint(0, len(label) - 1)
    if label[r] == 0:
        label[r] = 1
    else:
        label[r] = 0
#%%
plt.scatter(ages, label, color='red', marker='*')
plt.show()
#%%
import numpy as np
from sklearn.linear_model import LinearRegression
#%%
# Predição com Regressão linear
model = LinearRegression()
model.fit(ages.reshape(-1, 1), label)
#%%
#y = mx + b
m = model.coef_[0]
b = model.intercept_
#%%
# Entendendo os Coeficientes da Reta
from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots()

# marking the x-axis and y-axis
axis = plt.axes(xlim=(0, 2), ylim=(-0.1, 2))

# initializing a line variable
line, = axis.plot([], [], lw=3)

# data witch the line will contain (x, y)
def init():
    line.set_data([], [])
    return line,

def animate(i):
    m_copy = i * 0.01
    plt.title("m = " + str(m_copy))
    x = np.arange(0.0, 10.0, 0.1)
    y = m_copy * x + b
    line.set_data(x, y)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)
anim.save('linear_regression.gif', writer='imagemagick', fps=30)
#%%
# Entendendo os Coeficientes da Reta
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

# marking the x-axis and y-axis
axis = plt.axes(xlim=(0, 2), ylim=(-0.1, 2))

# initializing a line variable
line, = axis.plot([], [], lw=3)

# data witch the line will contain (x, y)
def init():
    line.set_data([], [])
    return line,

def animate(i):
    b_copy = i * 0.01
    plt.title("b = " + str(b_copy))
    x = np.arange(0.0, 10.0, 0.1)
    y = m * x + b_copy
    line.set_data(x, y)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                     frames=200, interval=20, blit=True)
anim.save('linear_regression.gif', writer='imagemagick', fps=30)
#%%
# Regressão Linear do Conjunto de Pontos
"""
0.5 = m * x + b
0.5 - b = m * x
(0.5 - b)/m = x
"""
limiar_idade = (0.5 - b) / m
plt.plot(ages, ages * m + b, color='green')
plt.plot([limiar_idade, limiar_idade], [0, 0.5], '--', color='blue')
plt.scatter(ages, label, color='red', marker='*')
plt.show()
#%%
# Função Logística
"""
y = 1/(1 + e^(-x))
"""
import math

def sigmoid(x):
    a = []
    for item in x:
        a.append(1 / (1 + math.exp(-item)))
    return a
#%%

x = np.arange(-10., 10., 0.2)
print(x)
#%%
sig = sigmoid(x)
print(sig)
#%%
plt.plot(x, sig)
plt.show()
#%%
# Classificador Sigmóide
"""
y = 1/(1 + e^(-z)) onde, z = m*x + b
"""
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(ages.reshape(-1, 1), label)

# y = m*x + b
m = model.coef_[0][0]
b = model.intercept_[0]

x = np.arange(0, 70, 0.1)
sig = sigmoid(x * m + b)

limiar_idade = abs(b/m)

plt.scatter(ages, label, color='red', marker='*')
plt.plot([limiar_idade, limiar_idade], [0, 0.5], '--', color='blue')
plt.plot(x, sig, color='green')
plt.show()
#%%
"""
0,5 = 1/(1 + e^(-(m*x + b)))
1 + e^(-(m*x + b)) = 1/0,5 = 2
e^(-(m*x + b)) = 2 - 1 = 1
- log(1)e = m*x + b
0 = mx + b
x = b/m 
"""
print(limiar_idade)
#%%
