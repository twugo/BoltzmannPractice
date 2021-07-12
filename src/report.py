# coding: utf-8

# writer: Takuya Togo

from itertools import repeat
import numpy as np
import matplotlib.pyplot as plt
import random
from boltzmann import *

class Printer:
    @staticmethod
    def plot_figure(data, title, filename = "out.png", xlabel="iteration", ylabel="E"):
        fig = plt.figure()
        plt.title(title)
        # plt.scatter(range(1, len(data)+1), data, s = 1, marker=".", )
        plt.plot(range(1, len(data)+1), data)
        

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(filename)

    @staticmethod
    def scatter_figure(data, title, filename = "out.png", xlabel="iteration", ylabel="E"):
        fig = plt.figure()
        plt.title(title)
        plt.scatter(range(1, len(data)+1), data, s = 1, marker=".", )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(filename)

class Solver:
    def excitement_only(self, n, weight_make_method, repeat_time):
        model = BoltzmannMachine(n, weight_make_method)
        # states = [0] * 2**n
        states = np.zeros(2**n)
        
        for i in range(repeat_time):
            model.excitement_dynamics()
            state = model.get_current_state()
            index = Calculater.pattern_to_index(state)

            states[index] += 1
        
        print(model.calc_state_probabilities())
        print(states / repeat_time)

        return states

    # def learn(self, n, weight_make_method, input_maker, repeat_time)

class Report:
    def __init__(self, seed=0):
        random.seed(seed)

    def q1(self):
        print("1-a")
        repeat_time = 1000000
        Solver().excitement_only(3, BoltzmannWeightMaker().zeros, repeat_time)

        print("1-b")
        l = 1000
        Solver().excitement_only(3, BoltzmannWeightMaker().uniform, l)
        l = 1000000
        Solver().excitement_only(3, BoltzmannWeightMaker().uniform, l)
        l = 10000000
        Solver().excitement_only(3, BoltzmannWeightMaker().uniform, l)

        print("1-c")
        l = 1000
        Solver().excitement_only(3, BoltzmannWeightMaker().pattern3and6only, l)
        l = 1000000
        Solver().excitement_only(3, BoltzmannWeightMaker().pattern3and6only, l)
    
    def q2_b(self):
        repeat_time = 100
        n = 3
        f = np.zeros((n+1, n+1))
        inputMaker = InputMakerFromProbabilityDistribution()

        for t in range(repeat_time):
            x = inputMaker.make(n)
            # print(x)

            for i in range(n+1):
                for j in range(i+1, n+1):
                    if(x[i] * x[j] == 1):
                        f[i][j] += 1

        for i in range(n+1):
            for j in range(i+1, n+1):
                print("f", i, j, "=", (int)(f[i][j]))
    
    def q2_d(self):
        repeat_time = 100
        n = 3
        g = np.zeros((n+1, n+1))

        model = BoltzmannMachine(n, BoltzmannWeightMaker().zeros)

        for t in range(repeat_time):
            model.excitement_dynamics()
            x = np.zeros(n+1)
            x[0] = 1
            x[1:] = model.get_current_state()
            
            for i in range(n+1):
                for j in range(i+1, n+1):
                    if(x[i] * x[j] == 1):
                        g[i][j] += 1

        g /= repeat_time

        for i in range(n+1):
            for j in range(i+1, n+1):
                print("g", i, j, "=", g[i][j])

    def q2_h(self):
        repeat_time = 10000
        n = 3
        q = [0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.4, 0.1]
        inputMaker = InputMakerFromProbabilityDistribution(q)

        model = BoltzmannMachine(n, BoltzmannWeightMaker().zeros, inputMaker)

        points = np.zeros(repeat_time)
        for t in range(repeat_time):
            model.learn_hebb_step()
            p = model.calc_state_probabilities()
            points[t] = Calculater.kullback_leibler_divergence(n, q, p)

        Printer.plot_figure(points, "q2_h", filename="q2_h.png", xlabel="t", ylabel="D")
        print(model._w)

    def q2_j_mini(self):
        repeat_time = 10000
        n = 3
        q = [0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.4, 0.1]
        inputMaker = InputMakerFromProbabilityDistribution(q)

        model = BoltzmannWithoutThreshold(n, BoltzmannWeightMaker().zeros, inputMaker)

        points = np.zeros(repeat_time)
        for t in range(repeat_time):
            model.learn_hebb_step()
            p = model.calc_state_probabilities()
            points[t] = Calculater.kullback_leibler_divergence(n, q, p)

        Printer.plot_figure(points, "q2_j", filename="q2_j.png", xlabel="t", ylabel="D")
        print(model._w)

    def q2_j(self):
        repeat_time = 10000
        n = 3
        q = [0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.4, 0.1]
        inputMaker = InputMakerFromProbabilityDistribution(q)

        model_h = BoltzmannMachine(n, BoltzmannWeightMaker().zeros, inputMaker)
        points_h = np.zeros(repeat_time)
        for t in range(repeat_time):
            model_h.learn_hebb_step()
            p = model_h.calc_state_probabilities()
            points_h[t] = Calculater.kullback_leibler_divergence(n, q, p)


        model_j = BoltzmannWithoutThreshold(n, BoltzmannWeightMaker().zeros, inputMaker)
        points_j = np.zeros(repeat_time)
        for t in range(repeat_time):
            model_j.learn_hebb_step()
            p = model_j.calc_state_probabilities()
            points_j[t] = Calculater.kullback_leibler_divergence(n, q, p)

        # plot
        fig = plt.figure()
        # plt.title('graph')
        plt.plot(range(1, len(points_j)+1), points_j, label='j')
        plt.plot(range(1, len(points_h)+1), points_h, label='h')
        plt.legend()

        plt.xlabel('t')
        plt.ylabel('D')
        plt.savefig('q2_j_h_graph')

    def q3(self):
        '''
        閾値ありの学習済みの回路を x_1=1 だけ固定し、更新
        '''
        print('q3')
        # 閾値ありで学習済みの回路を作る
        learn_time = 10000 # 学習回数
        n = 3
        q = [0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.4, 0.1]
        inputMaker = InputMakerFromProbabilityDistribution(q)

        model = BoltzmannMachine(n, BoltzmannWeightMaker().zeros, inputMaker)

        points = np.zeros(learn_time)
        for t in range(learn_time):
            model.learn_hebb_step()
            p = model.calc_state_probabilities()
            points[t] = Calculater.kullback_leibler_divergence(n, q, p)

        Printer.plot_figure(points, '', filename="q3_graph.png", xlabel="t", ylabel="D")
        print(model._w)

        # x_1=1と固定し更新、出現値から確率計算
        repeat_time = 10000
        states = [0] * 2**n
        
        for i in range(repeat_time):
            model.excitement_dynamics_fixing_neuron()
            tmp = model.get_current_state()
            index = 0
            for i in range(n):
                index += tmp[i] * 2**i
            index = (int)(index)

            states[index] += 1
        
        print(model.calc_state_probabilities())
        print(states)

        return states

class BoltzmannWithoutThreshold(BoltzmannMachine):
    def __init__(self, n, weight_make_method, input_maker=InputMakerFromProbabilityDistribution()):
        super().__init__(n, weight_make_method, input_maker=InputMakerFromProbabilityDistribution())
    
    def learn_hebb_step(self):
        repeat_time = 100
        n = 3

        g = np.zeros((n+1, n+1))

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
        
        for i in range(n+1):
            for j in range(n+1):
                if(i == j or i == 0 or j == 0):
                    self._w[i][j] = 0
                else:
                    if i<j: # 上三角の時
                        self._w[i][j] += 0.01 * (self._f[i][j] - g[i][j])
                    else:
                        self._w[i][j] = self._w[j][i] # 上三角の方が先に計算されているはずである

def main():
    report = Report()
    report.q1()
    # report.q1()


if __name__ == '__main__':
    main()
