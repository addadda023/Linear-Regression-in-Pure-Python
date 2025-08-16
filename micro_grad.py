"""Micrograd with support for custom optimizers."""

import math
from typing import Tuple, Self

class Value:
    def __init__(self, value: float, _children: Tuple[Self, Self]=(), _op: str='', label: str=''):
        self.value = value
        self.grad = 0.0
        self._backward = lambda: None
        self.children = set(_children)
        self.op = _op
        self.label = label
    
    def __repr__(self):
        return(f"Value(value={self.value})")
    
    def __add__(self, other) -> Self:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value + other.value, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other) -> Self:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value * other.value, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value
        out._backward = _backward
        
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.value**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.value**(other - 1)) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

class Optimizer:
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
