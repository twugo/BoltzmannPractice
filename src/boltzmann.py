# coding: utf-8

# writer: Takuya Togo

from itertools import repeat
import numpy as np
import math
import random
from abc import ABC, ABCMeta, abstractmethod

class IInputMaker(ABC):
    '''
    入力信号作成インターフェース
    '''
    @abstractmethod
    def __init__(self, seed=0):
        pass

    @abstractmethod
    def make(self, n):
        pass

class InputMakerFromProbabilityDistribution(IInputMaker):
    def __init__(self, probability=[0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.4, 0.1], seed=0):
        self._rng_generator = np.random.default_rng(seed)
        self.probability = probability
    
    def make(self, n):
        pattern =  self._rng_generator.choice(list(range(2**n)), p=self.probability)

        x = Calculater.index_to_pattern(pattern, n)
        return x

class Calculater:
    @staticmethod
    def index_to_pattern(index, n):
        # [x_0=1, x_1, x_2, x_3, ...]

        x = np.zeros(n+1)
        tmp = index
        for i in range(n):
            x[n-i] = (int)(tmp % 2)
            tmp = (int)(tmp/2)
        x[0] = 1

        return x

    @staticmethod
    def pattern_to_index(pattern):
        '''
        パターンを渡すとインデックスに変換する
        Args:
            x_0を除いたパターン
        '''
        n = len(pattern)
        index = 0
        for i in range(len(pattern)):
            index += pattern[n-1-i] * 2**i
        index = (int)(index)

        return index


    @staticmethod
    def kullback_leibler_divergence(n, q, p):
        divergence = 0

        for i in range(2**n):
            divergence += q[i] * math.log(q[i]/p[i])

        return divergence


class BoltzmannMachine:
    def __init__(self, n, weight_make_method, input_maker=InputMakerFromProbabilityDistribution()):
        self._x = np.zeros(n+1)
        self._x[0] = 1
        self._w = weight_make_method(n)

        self._T = 1 # 温度パラメータ、
        self._f = self._calc_f(input_maker) # ヘブ学習に使用する値
    
    def _calc_f(self, input_maker, repeat_time=100):
        '''ヘブ学習に使用するfを計算する
        '''
        n = len(self._x) - 1

        f = np.zeros((n+1, n+1))

        for t in range(repeat_time):
            x = input_maker.make(n)

            for i in range(n+1):
                for j in range(i+1, n+1):
                    if(x[i] * x[j] == 1):
                        f[i][j] += 1

        f /= repeat_time

        return f

    def excitement_dynamics(self):
        '''
        興奮のダイナミクス
        ランダムに1つニューロンを選び、計算した確率以上か以下かでさらにランダムに0,1を決める
        '''
        n = len(self._x) - 1
        rand_num = random.randint(1, n)

        u = 0
        for j in range(n+1):
            u += self._w[rand_num][j] * self._x[j]
        
        Pr1 = 1 / (1 + math.exp(-u / self._T))
        if(random.random() < Pr1):
            self._x[rand_num] = 1
        else:
            self._x[rand_num] = 0

    def excitement_dynamics_fixing_neuron(self, free_neuron=[2,3]):
        '''
        興奮のダイナミクス
        特定のニューロンは固定
        ランダムに1つニューロンを選び、計算した確率以上か以下かでさらにランダムに0,1を決める
        '''
        n = len(self._x) - 1
        rand_num = random.choice(free_neuron)

        u = 0
        for j in range(n+1):
            u += self._w[rand_num][j] * self._x[j]
        
        Pr1 = 1 / (1 + math.exp(-u / self._T))
        if(random.random() < Pr1):
            self._x[rand_num] = 1
        else:
            self._x[rand_num] = 0
    
    def learn_hebb_step(self):
        n = len(self._x)-1

        # gの計算
        g = np.zeros((n+1, n+1))

        repeat_time = 100
        for t in range(repeat_time):
            self.excitement_dynamics()
            x = np.zeros(n+1)
            x[0] = 1
            x[1:] = self.get_current_state()
            
            for i in range(n+1):
                for j in range(i+1, n+1):
                    if(x[i] * x[j] == 1):
                        g[i][j] += 1

        g /= repeat_time
        
        # 結合係数の値を更新
        for i in range(n+1):
            for j in range(n+1):
                if(i == j): 
                    self._w[i][j] = 0
                else:
                    if i<j: # 上三角の時
                        self._w[i][j] += 0.01 * (self._f[i][j] - g[i][j])
                    else:
                        self._w[i][j] = self._w[j][i] # 上三角の方が先に計算されているはずである
    
    def learn_hebb_step_fixing_neuron(self):
        n = len(self._x)-1

        # gの計算
        g = np.zeros((n+1, n+1))

        repeat_time = 100
        for t in range(repeat_time):
            self.excitement_dynamics_fixing_neuron(free_neuron=[2,3])
            x = np.zeros(n+1)
            x[0] = 1
            x[1:] = self.get_current_state()
            
            for i in range(n+1):
                for j in range(i+1, n+1):
                    if(x[i] * x[j] == 1):
                        g[i][j] += 1

        g /= repeat_time

        # 結合係数の値を更新
        for i in range(n+1):
            for j in range(n+1):
                if(i == j): 
                    self._w[i][j] = 0
                else:
                    if i<j: # 上三角の時
                        self._w[i][j] += 0.01 * (self._f[i][j] - g[i][j])
                    else:
                        self._w[i][j] = self._w[j][i] # 上三角の方が先に計算されているはずである


    def calc_state_probabilities(self):
        n = len(self._x) - 1
        probabilities = np.zeros(2**n)

        for pattern_index in range(2**n):
            x = Calculater.index_to_pattern(pattern_index, n)

            # Eの計算
            E = 0
            for i in range(n+1):
                for j in range(i+1, n+1):
                    E -= self._w[i][j] * x[i] * x[j]
            
            probabilities[pattern_index] = math.exp(-E / self._T)
        
        c_denominator = np.sum(probabilities)

        probabilities /= c_denominator

        return probabilities
        
        
    def get_current_state(self):
        return self._x[1:]

class BoltzmannWeightMaker:
    def __init__(self, seed=0):
        self._rng_generator = np.random.default_rng(seed)

    def zeros(self, n):
        return np.zeros((n+1, n+1))
    
    def uniform(self, n, min=-5, max=5):
        w = self._rng_generator.random((n+1, n+1)) * (max-min) + min
        for i in range(n+1):
            w[i][i] = 0 # 対角成分を0にする
            for j in range(i+1, n+1):
                w[j][i] = w[i][j] # 対称行列にする
        
        return w
    
    def pattern3and6only(self, n):
        # w02=w20 = 2, w13=w31 = -2, それ以外0
        w = np.zeros((n+1, n+1))
        w[0][2] = 2
        w[2][0] = 2
        w[1][3] = -2
        w[3][1] = -2
        
        return w

if __name__ == '__main__':
    random.seed(0)
