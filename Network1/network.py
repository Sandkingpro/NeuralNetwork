"""""""""""""""""""""""""""""""""""""""
network.py
Модуль создания и обучения нейронной сети для распознавания рукописных цифр
с использованием метода градиентного спуска.
Группа:КЭ-128
ФИО:Дремин М.А.
"""""""""""""""""""""""""""""""""""""""
#### Библиотеки
# Стандартные библиотеки
import random  # библиотека функций для генерации случайных значений
# Сторонние библиотеки
import numpy as np  # библиотека функций для работы с матрицами

""" ---Раздел описаний--- """
""" --Описание класса Network--"""


class Network(object):  # используется для описания нейронной сети
    def __init__(self, sizes):  # конструктор класса
        # self – указатель на объект класса
        # sizes – список размеров слоев нейронной сети
        self.num_layers = len(sizes)  # задаем количество слоев нейронной\ сети
        self.sizes = sizes  # задаем список размеров слоев нейронной сети
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # задаем случайные начальные смещения
        self.weights = [np.random.randn(y, x) for x, y in
                        zip(sizes[:-1], sizes[1:])]  # задаем случайные начальные веса связей

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(  # Стохастический градиентный спуск
            self  # указатель на объект класса
            , training_data  # обучающая выборка
            , epochs  # количество эпох обучения
            , mini_batch_size  # размер подвыборки
            , eta  # скорость обучения
            , test_data  # тестирующая выборка
    ):
        test_data = list(test_data)  # создаем список объектов тестирующей выборки
        n_test = len(test_data)  # вычисляем длину тестирующей выборки
        training_data = list(training_data)  # создаем список объектов обучающей выборки
        n = len(training_data)  # вычисляем размер обучающей выборки
        for j in range(epochs):  # цикл по эпохам
            random.shuffle(training_data)  # перемешиваем элементы обучающей выборки
            mini_batches = [training_data[k:k + mini_batch_size] for k in
                            range(0, n, mini_batch_size)]  # создаем подвыборки
            for mini_batch in mini_batches:  # цикл по подвыборкам
                self.update_mini_batch(mini_batch, eta)  # один шаг градиентного спуска
            print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))  # смотрим прогресс в обучении

    def update_mini_batch(  # Шаг градиентного спуска
            self  # указатель на объект класса
            , mini_batch  # подвыборка
            , eta  # скорость обучения
    ):
        nabla_b = [np.zeros(b.shape) for b in
                   self.biases]  # список градиентов dC/db для каждого слоя (первоначально заполняются нулями)
        nabla_w = [np.zeros(w.shape) for w in
                   self.weights]  # список градиентов dC/dw для каждого слоя (первоначально заполняются нулями)
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,
                                                         y)  # послойно вычисляем градиенты dC/db и dC/dw для текущего прецедента (x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b,
                                                   delta_nabla_b)]  # суммируем градиенты dC/db для различных прецедентов текущей подвыборки
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w,
                                                   delta_nabla_w)]  # суммируем градиенты dC/dw для различных прецедентов текущей подвыборки
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]  # обновляем все веса w нейронной сети
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]  # обновляем все смещения b нейронной сети

    def backprop(  # Алгоритм обратного распространения
            self  # указатель на объект класса
            , x  # вектор входных сигналов
            , y  # ожидаемый вектор выходных сигналов
    ):
        nabla_b = [np.zeros(b.shape) for b in
                   self.biases]  # список градиентов dC/db для каждого слоя (первоначально заполняются нулями)
        nabla_w = [np.zeros(w.shape) for w in
                   self.weights]  # список градиентов dC/dw для каждого слоя (первоначально заполняются нулями)
        # определение переменных
        activation = x  # выходные сигналы слоя (первоначально соответствует выходным сигналам 1-го слоя или входным сигналам сети)
        activations = [
            x]  # список выходных сигналов по всем слоям (первоначально содержит только выходные сигналы 1-го слоя)
        zs = []  # список активационных потенциалов по всем слоям (первоначально пуст)

        # прямое распространение
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b  # считаем активационные потенциалы текущего слоя
            zs.append(z)  # добавляем элемент (активационные потенциалы слоя) в конец списка
            activation = sigmoid(
                z)  # считаем выходные сигналы текущего слоя, применяя сигмоидальную функцию активации к активационным потенциалам слоя
            activations.append(activation)  # добавляем элемент (выходные сигналы слоя) в конец списка

        # обратное распространение
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(
            zs[-1])  # считаем меру влияния нейронов выходного слоя L на величину ошибки (BP1)
        nabla_b[-1] = delta  # градиент dC/db для слоя L (BP3)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # градиент dC/dw для слоя L (BP4)

        for l in range(2, self.num_layers):
            z = zs[-l]  # активационные потенциалы l-го слоя (двигаемся по списку справа налево)
            sp = sigmoid_prime(z)  # считаем сигмоидальную функцию от активационных потенциалов l-го слоя
            delta = np.dot(self.weights[-l + 1].transpose(),
                           delta) * sp  # считаем меру влияния нейронов l-го слоя на величину ошибки (BP2)
            nabla_b[-l] = delta  # градиент dC/db для l-го слоя (BP3)
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())  # градиент dC/dw для l-го слоя (BP4)
        return nabla_b, nabla_w

    def evaluate(self, test_data):  # Оценка прогресса в обучении
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations,
                        y):  # Вычисление частных производных стоимостной функции по выходным сигналам последнего слоя
        return (output_activations - y)


def sigmoid(z):  # определение сигмоидальной функции активации
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):  # Производная сигмоидальной функции
    return sigmoid(z) * (1 - sigmoid(z))
