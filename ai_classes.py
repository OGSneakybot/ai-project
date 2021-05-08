import random as rand
import tensorflow as tf


def format_random_weight():
    weight = float(format(rand.randint(-100, 100) * 0.1, ".2f"))
    return weight


def format_random_bias():
    bias = float(format(rand.randint(-10, 10), ".2f"))
    return bias


class Loop:
    def __init__(self, fitness_copy, w_copy, b_copy, training_data):
        self.fitness_copy = fitness_copy
        self.w_copy = w_copy
        self.b_copy = b_copy
        self.training_data = training_data

    def train(self, r1_copy, fitness_copy, w_copy, b_copy):
        for _ in self.training_data:
            i1_copy = float(self.training_data[r1_copy][0])
            i2_copy = float(self.training_data[r1_copy][1])
            i3_copy = float(self.training_data[r1_copy][2])
            print("input 1 =", i1_copy)
            print("input 2 =", i2_copy)
            print("input 3 =", i3_copy)
            o1_copy = (i1_copy * w_copy[0]) + (i2_copy * w_copy[1]) + (i3_copy * w_copy[2]) + b_copy[0]
            o2_copy = (i1_copy * w_copy[3]) + (i2_copy * w_copy[4]) + (i3_copy * w_copy[5]) + b_copy[1]
            o1_copy = format(tf.sigmoid(o1_copy), ".2f")
            o2_copy = format(tf.sigmoid(o2_copy), ".2f")
            if o1_copy > o2_copy:
                guess_copy = "apple"
                print("The ai guessed apple")
            elif o2_copy > o1_copy:
                guess_copy = "orange"
                print("The ai guessed orange")
            else:
                guess_copy = ""
                print("The ai guessed nothing")
            print("the ai should guess", self.training_data[r1_copy][3])
            print("output1 =", o1_copy, "\n" + "output2 =", o2_copy)
            if guess_copy == self.training_data[r1_copy][3]:
                fitness_copy += 1
                print("\n")
            r1_copy += 1
            self.fitness_copy = fitness_copy
            self.w_copy = w_copy
            self.b_copy = b_copy

    def get_fitness(self):
        return self.fitness_copy

    def get_w_copy(self):
        return self.w_copy

    def get_b_copy(self):
        return self.b_copy
