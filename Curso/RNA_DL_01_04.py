# Redes Neurais e Deep Learning 01 e 02
#%%
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
# Redes Neurais e Deep Learning 03
"""
Perceptron
x1  --> w1  --v
x2  --> w2  --> Wo  --> f(z) = {1, se z > 0} ou {0, se z <= 0}
x3  --> w3  --^

f(z) = w1 * x1 + w2 * x2 + w3 * x3 + Wo

wj = wj + D_wj
D_wj = l * (yi - yi´) * Xj
"""
#%%
# Redes Neurais e Deep Learning 04
"""
Função de custo
j(w) = 1/2 * sum(yi - yi´)^2

y2 - y1 = m(x2 - x1)
tg0 = co/ca = y2 - y1 / x2 - x1

m = (y2 - y1) / (x2 - x1)
m = f(x2) - f(x1) / x2 - x1
m = f(x1 + Dx) - f(x1) / x1 + Dx - x1
m = f(x1 + Dx) - f(x1) / Dx

lim Dx->0 = f(x1 + Dx) - f(x1) / Dx

j(w) = 1/2 * sum(yi - f(z))^2
f(z) = w1 * x1 + w2 * x2 + w3 * x3 + Wo

Derivado em função de (w):
dj/dwj = 1/2 * dsum(yi - f(z))^2
dj/dwj = 1/2 * 2 * dsum(yi - f(z)) * d(yi - (w1 * x1 + w2 * x2 + w3 * x3 + Wo))
dj/dwj = sum(yi - f(z)) * (-xj)
dj/dwj = - sum(yi - f(z)) * (xj)

D_wj = l * sum(yi - f(z)) * (xj)
"""
#%%
# Perceptron Simples

# Importação de bibliotecas (se necessário)

# Definição de variáveis e constantes

# Dados de treinamento (substitua por seus dados reais)
data = [
    [0, 0, 1, 1],  # Exemplo 1
    [1, 0, 1, 1],  # Exemplo 2
    [1, 1, 1, 0],  # Exemplo 3
    [1, 1, 0, 0]   # Exemplo 4
]

# Pesos iniciais (substitua por valores iniciais desejados)
w = [0, 0, 0, 0]

# Índices das features e do rótulo
X_1 = 1
X_2 = 2
X_3 = 3
Y = 3

# Índice do exemplo atual (inicializado em 0)
SAMPLE = 0

# Taxa de aprendizado (controla a magnitude das atualizações dos pesos)
learning_rate = 0.1

# Número de épocas de treinamento
num_epochs = 10

# Loop principal de treinamento
for epoch in range(num_epochs):
    # Loop para cada exemplo nos dados de treinamento
    for SAMPLE in range(len(data)):
        # Cálculo da combinação linear (soma ponderada das features)
        z = w[0] + w[1] * data[SAMPLE][X_1] + w[2] * data[SAMPLE][X_2] + w[3] * data[SAMPLE][X_3]

        # Predição do rótulo
        if z > 0:
            pred_y = 1
        else:
            pred_y = 0

        # Atualização dos pesos usando a regra de aprendizado do Perceptron
        w[1] = w[1] + learning_rate * (data[SAMPLE][Y] - pred_y) * data[SAMPLE][X_1]
        w[2] = w[2] + learning_rate * (data[SAMPLE][Y] - pred_y) * data[SAMPLE][X_2]
        w[3] = w[3] + learning_rate * (data[SAMPLE][Y] - pred_y) * data[SAMPLE][X_3]
        w[0] = w[0] + learning_rate * (data[SAMPLE][Y] - pred_y)

        # Impressão da combinação linear e dos pesos atualizados
        print(f"Combinação linear para o exemplo {SAMPLE + 1}: {z}")
        print(f"Pesos atualizados após o exemplo {SAMPLE + 1}: {w}")

    # Avançar para o próximo exemplo no próximo ciclo da época
    SAMPLE = (SAMPLE + 1) % len(data)

print("Treinamento concluído!")

#%%
w[0] + w[1] * 1 + w[2] * 1 + w[3] * 1
#%%
# Adaline

# Importação de bibliotecas (se necessário)
# ... (Consider importing libraries for visualization if needed)

# Definição de dados de treinamento (substitua por seus dados reais)
data = [[0, 0, 1, 1],  # Exemplo 1
        [1, 0, 1, 1],  # Exemplo 2
        [1, 1, 1, 0],  # Exemplo 3
        [1, 1, 0, 0]]  # Exemplo 4

# Inicialização de pesos (substitua por valores iniciais desejados se necessário)
w = [0, 0, 0, 0]

# Índices das features e do rótulo
X_1 = 0
X_2 = 1
X_3 = 2
Y = 3

# Taxa de aprendizado (controla a magnitude das atualizações dos pesos)
learning_rate = 0.1

# Função para calcular a combinação linear (soma ponderada das features)
def z(sample):
    """
    Calcula a combinação linear para um determinado exemplo de dados.

    Args:
        sample (int): Índice do exemplo de dados no conjunto de treinamento.

    Returns:
        float: Valor da combinação linear para o exemplo especificado.
    """
    return w[0] + w[1] * data[sample][X_1] + w[2] * data[sample][X_2] + w[3] * data[sample][X_3]

# Função de ativação degrau (passo a passo)
def phi(value):
    """
    Função de ativação degrau (passo a passo) para classificação binária.

    Args:
        value (float): Valor da combinação linear.

    Returns:
        int: 1 se o valor for positivo, 0 caso contrário.
    """
    if value > 0:
        return 1
    else:
        return 0

# Função para calcular a função de custo (erro médio quadrático)
def J():
    """
    Calcula a função de custo (erro médio quadrático) para t0do o conjunto de treinamento.

    Returns:
        float: Valor da função de custo.
    """
    error_sum = 0
    for sample in range(len(data)):
        error_sum += (data[sample][-1] - phi(z(sample)))**2  # Erro quadrático para cada exemplo
    return 1/2 * error_sum  # Média dos erros quadráticos

# Primeira Época
print("--- 1ª Época ---")
print("Erro J(w) =", J())

# Função para calcular a derivada da função de custo em relação a um peso específico
def delta_j(j):
    """
    Calcula a derivada parcial da função de custo em relação a um peso específico.

    Args:
        j (int): Índice do peso (0 para bias, 1 para a primeira feature, etc.).

    Returns:
        float: Valor da derivada parcial.
    """
    error_sum = 0
    for sample in range(len(data)):
        error = data[sample][-1] - phi(z(sample))  # Erro para o exemplo atual
        error_sum += error * data[sample][j]  # Soma ponderada do erro por feature
    return learning_rate * error_sum

# Atualização dos pesos
aux_0 = w[0] + delta_j(0)
aux_1 = w[1] + delta_j(1)
aux_2 = w[2] + delta_j(2)
aux_3 = w[3] + delta_j(3)
w = [aux_0, aux_1, aux_2, aux_3]

print("w =", w)

#%%
# Treinamento por 5 épocas

epochs = 5
for i in range(epochs):
    print("---",i+2,"º Época---")
    print("Erro J(w) =", J())

    aux_0 = w[0] + delta_j(0)
    aux_1 = w[1] + delta_j(1)
    aux_2 = w[2] + delta_j(2)
    aux_3 = w[3] + delta_j(3)

    w = [aux_0, aux_1, aux_2, aux_3]
    print("w =", w)
    print("\n")
#%%
# Adaline com base de dados de Diabetes

import pandas as pd

data = pd.read_csv(r"D:\Users\Nayan Couto\Cloud Drive\Documentos\Arquivos PDF, PPT, DOC\Ciências de Dados - Anhanguera Ampli\Inteligência Artificial\Redes Neurais e Deep Learning\RNA_Deep_Learning\Curso\diabetes.csv").values

data.shape[0]
#%%
w = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

X = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]

SAMPLE = 0
learning_rate = 0.15

def z(S):

    """
    Calcula a combinação linear para uma dada amostra usando os pesos e recursos.

    Parâmetros:
        S (int): O índice da amostra.

    Retorna:
        float: A combinação linear para a amostra dada.
    """

    return w[0] + w[1] * data[S][X[1]] + w[2] * data[S][X[2]] + w[3] * data[S][X[3]] + w[4] * data[S][X[4]] + \
        w[5] * data[S][X[5]] + w[6] * data[S][X[6]] + w[7] * data[S][X[7]] + w[8] * data[S][X[8]]

def output(z):

    """
    Calcula a saída de uma rede neural com base no valor de entrada z.

    Parâmetros:
        z (float): O valor de entrada para a rede neural.

    Retorna:
        int: 1 se o valor de entrada for maior que 0, 0 caso contrário.
    """

    if z > 0:
        return 1
    else:
        return 0

def J():

    """
    Calcula o valor da função de custo J para um determinado conjunto de dados.

    Retorna:
        float: O valor calculado da função de custo J.
    """

    sum_errors = 0
    for i in range(data.shape[0]):
        sum_errors += (data[i][-1] - output(z(i)))**2
    return 1/2 * sum_errors
    #return 1/2 * ((data[0][-1] - output(z(0)))**2 + (data[1][-1] - output(z(1)))**2 + (data[2][-1] - output(z(2)))**2 + (data[3][-1] - output(z(3)))**2)

def acc():

    """
    Calcula a acurácia do modelo comparando a saída prevista com a etiqueta real.

    Retorna:
        float: A acurácia do modelo, calculada como a razão de rótulos previstos corretamente dividido pelo número total de rótulos.
    """

    count_hits = 0
    for i in range(data.shape[0]):
        if int(data[i][-1]) == output(z(i)):
            count_hits += 1
    return count_hits / data.shape[0]

print("J = ", J())

def delta_j(j):

    """
    Calcula o valor delta para a unidade de saída especificada em uma rede neural.

    Parâmetros:
        j (int): O índice da unidade de saída.

    Retorna:
        float: O valor delta para a unidade de saída especificada.
    """

    if j == 0:
        aux = 0
        for i in range(data.shape[0]):
            aux += data[i][-1] - output(z(i))

        return learning_rate * aux
    else:
        aux = 0
        for i in range(data.shape[0]):
            aux += (data[i][-1] - output(z(i))) * data[i][X[j]]

        return learning_rate * aux

aux_0 = w[0] + delta_j(0)
aux_1 = w[1] + delta_j(1)
aux_2 = w[2] + delta_j(2)
aux_3 = w[3] + delta_j(3)
aux_4 = w[4] + delta_j(4)
aux_5 = w[5] + delta_j(5)
aux_6 = w[6] + delta_j(6)
aux_7 = w[7] + delta_j(7)
aux_8 = w[8] + delta_j(8)

w = [aux_0, aux_1, aux_2, aux_3, aux_4, aux_5, aux_6, aux_7, aux_8]

print("J = ", J(), "acc = ", acc(), "w = ", w) # w
#%%
# Treinamento Contínu
i = 0
acc_aux = 0

while 1:
    if i % 1000 == 0:
        print(".")

    if acc() > acc_aux:
        acc_aux = acc()
        print(J(), acc(), w)

    aux_0 = w[0] + delta_j(0)
    aux_1 = w[1] + delta_j(1)
    aux_2 = w[2] + delta_j(2)
    aux_3 = w[3] + delta_j(3)
    aux_4 = w[4] + delta_j(4)
    aux_5 = w[5] + delta_j(5)
    aux_6 = w[6] + delta_j(6)
    aux_7 = w[7] + delta_j(7)
    aux_8 = w[8] + delta_j(8)

    w = [aux_0, aux_1, aux_2, aux_3, aux_4, aux_5, aux_6, aux_7, aux_8]

    w

    i += 1
#%%
