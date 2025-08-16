import math


class Optimizer:
    """Base class for Optimizer."""

    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0


class AdagradOptimizer(Optimizer):
    def __init__(self, parameters, learning_rate=0.01):
        super().__init__(parameters, learning_rate)
        self.cache = {p: 0 for p in self.parameters}

    def step(self):
        for p in self.parameters:
            self.cache[p] += p.grad**2
            p.value -= self.learning_rate * p.grad / (math.sqrt(self.cache[p]) + 1e-8)

class SGDOptimizer(Optimizer):
    def __init__(self, parameters, learning_rate=0.01):
        super().__init__(parameters, learning_rate)

    def step(self):
        for p in self.parameters:
            p.value -= self.learning_rate * p.grad

class MomentumOptimizer(Optimizer):
    def __init__(self, parameters, learning_rate=0.01, momentum=0.9):
        super().__init__(parameters, learning_rate)
        self.momentum = momentum
        self.velocity = {p: 0 for p in self.parameters}

    def step(self):
        for p in self.parameters:
            self.velocity[p] = self.momentum * self.velocity[p] + p.grad
            p.value -= self.learning_rate * self.velocity[p]

class AdamOptimizer(Optimizer):
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(parameters, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {p: 0 for p in self.parameters}
        self.v = {p: 0 for p in self.parameters}
        self.t = 0

    def step(self):
        self.t += 1
        for p in self.parameters:
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * p.grad
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (p.grad**2)
            m_hat = self.m[p] / (1 - self.beta1**self.t)
            v_hat = self.v[p] / (1 - self.beta2**self.t)
            p.value -= self.learning_rate * m_hat / (math.sqrt(v_hat) + self.epsilon)
