import tensorflow as tf
from _tkinter import *

import pickle

from future.moves import tkinter

import ai_classes as classes


# main (root) GUI menu
class CrudGUI:
    def __init__(self, master):
        self.master = master
        self.master.title('Welcome Menu')

        self.top_frame = tkinter.Frame(self.master)
        self.bottom_frame = tkinter.Frame(self.master)

        self.radio_var = tkinter.IntVar()
        self.radio_var.set(1)

        # create the radio buttons
        self.look = tkinter.Radiobutton(self.top_frame, text='View my ai',
                                        variable=self.radio_var, value=1)
        self.add = tkinter.Radiobutton(self.top_frame, text='View your ai',
                                       variable=self.radio_var, value=2)
        self.change = tkinter.Radiobutton(self.top_frame, text='train ai',
                                          variable=self.radio_var, value=3)

        # pack the radio buttons
        self.look.pack(anchor='w', padx=20)
        self.add.pack(anchor='w', padx=20)
        self.change.pack(anchor='w', padx=20)

        # create ok and quit buttons
        self.ok_button = tkinter.Button(self.bottom_frame, text='OK', command=self.open_menu)
        self.quit_button = tkinter.Button(self.bottom_frame, text='QUIT', command=self.master.destroy)

        # pack the buttons
        self.ok_button.pack(side='left')
        self.quit_button.pack(side='left')

        # pack the frames
        self.top_frame.pack()
        self.bottom_frame.pack()

    def open_menu(self):
        if self.radio_var.get() == 1:
            _ = ViewMyAi(self.master)
        elif self.radio_var.get() == 2:
            _ = ViewYourAi(self.master)
        elif self.radio_var.get() == 3:
            _ = TrainAi(self.master)
        else:
            pass


class ViewMyAi:
    def __init__(self, master):
        self.my_weights = [7.9, 0.7, 3.6, -9.8, -5.1, -0.3]
        self.my_biases = [-7.3, 4.6]
        self.training_data = [[0.0, 0.0, 0.0, "orange"], [0.0, 0.0, 1.0, "orange"],[0.0, 1.0, 0.0, "orange"],
                              [1.0, 0.0, 1.0, "apple"], [1.0, 1.0, 0.0, "apple"], [1.0, 1.0, 1.0, "apple"]]
        self.fruit = 0
        self.o1 = (self.training_data[self.fruit][0] * self.my_weights[0]) + (self.training_data[self.fruit][1] * self.my_weights[1]) + (self.training_data[self.fruit][2] * self.my_weights[2]) + self.my_biases[0]
        self.o2 = (self.training_data[self.fruit][0] * self.my_weights[3]) + (self.training_data[self.fruit][1] * self.my_weights[4]) + (self.training_data[self.fruit][2] * self.my_weights[5]) + self.my_biases[1]
        self.o1 = format(tf.sigmoid(self.o1), ".2f")
        self.o2 = format(tf.sigmoid(self.o2), ".2f")
        if self.o1 > self.o2:
            self.ai_guess = "apple"
        elif self.o1 < self.o2:
            self.ai_guess = "orange"
        else:
            self.ai_guess = "none"

        # tkinter.Top_level() is like tkinter.Frame() but it opens in a new window
        self.look = tkinter.Toplevel(master)
        self.look.title('View my ai')

        # create Frames for this Top_level window
        self.top_frame = tkinter.Frame(self.look)
        self.middle_top_frame = tkinter.Frame(self.look)
        self.middle_bottom_frame = tkinter.Frame(self.look)
        self.bottom_frame = tkinter.Frame(self.look)

        # widgets for top frame - label and entry box for name
        self.inputs_text = tkinter.Label(self.top_frame, text=f'input 1 = {self.training_data[self.fruit][0]} input 2 = {self.training_data[self.fruit][2]} input 3 = {self.training_data[self.fruit][2]}')

        # pack top frame
        self.inputs_text.pack(side="left")

        # middle top frame label for weights and biases
        self.weight = tkinter.StringVar()
        self.weights_label = tkinter.Label(self.middle_top_frame, text=f"weights: {self.my_weights} biases: {self.my_biases}")

        # pack middle top frame
        self.weights_label.pack(side="left")

        # middle bottom frame - label for outputs
        self.outputs = tkinter.StringVar()
        self.result_label = tkinter.Label(self.middle_bottom_frame, text=f"Output 1: {self.o1} Output 2: {self.o2} \nAi should guess: {self.training_data[self.fruit][3]} Ai guess is: {self.ai_guess}")

        # pack middle bottom frame
        self.result_label.pack(side='left')

        # buttons for bottom frame
        self.menu_button = tkinter.Button(self.bottom_frame, text='Main Menu', command=self.menu)
        self.back_button = tkinter.Button(self.bottom_frame, text='back', command=self.back)
        self.next_button = tkinter.Button(self.bottom_frame, text='Next', command=self.next)

        # pack bottom frame
        self.menu_button.pack(side='left')
        self.back_button.pack(side='left')
        self.next_button.pack(side="left")

        # pack frames
        self.top_frame.pack()
        self.middle_top_frame.pack()
        self.middle_bottom_frame.pack()
        self.bottom_frame.pack()

    def next(self):
        if self.fruit <= 4:
            self.fruit += 1
            self.inputs_text.destroy()
            self.inputs_text = tkinter.Label(self.top_frame, text=f'input 1 = {self.training_data[self.fruit][0]} input 2 = {self.training_data[self.fruit][1]} input 3 = {self.training_data[self.fruit][2]}')
            self.inputs_text.pack(side="left")
            self.result_label.destroy()
            self.o1 = (self.training_data[self.fruit][0] * self.my_weights[0]) + (self.training_data[self.fruit][1] * self.my_weights[1]) + (self.training_data[self.fruit][2] * self.my_weights[2]) + self.my_biases[0]
            self.o2 = (self.training_data[self.fruit][0] * self.my_weights[3]) + (self.training_data[self.fruit][1] * self.my_weights[4]) + (self.training_data[self.fruit][2] * self.my_weights[5]) + self.my_biases[1]
            self.o1 = format(tf.sigmoid(self.o1), ".2f")
            self.o2 = format(tf.sigmoid(self.o2), ".2f")
            if self.o1 > self.o2:
                self.ai_guess = "apple"
            elif self.o1 < self.o2:
                self.ai_guess = "orange"
            else:
                self.ai_guess = "none"
            self.result_label = tkinter.Label(self.middle_bottom_frame, text=f"Output 1: {self.o1} Output 2: {self.o2} \nAi should guess: {self.training_data[self.fruit][3]} Ai guess is: {self.ai_guess}")
            self.result_label.pack(side="left")

    def back(self):
        if self.fruit >= 1:
            self.fruit -= 1
            self.inputs_text.destroy()
            self.inputs_text = tkinter.Label(self.top_frame, text=f'input 1 = {self.training_data[self.fruit][0]} input 2 = {self.training_data[self.fruit][1]} input 3 = {self.training_data[self.fruit][2]}')
            self.inputs_text.pack(side="left")
            self.result_label.destroy()
            self.o1 = (self.training_data[self.fruit][0] * self.my_weights[0]) + (self.training_data[self.fruit][1] * self.my_weights[1]) + (self.training_data[self.fruit][2] * self.my_weights[2]) + self.my_biases[0]
            self.o2 = (self.training_data[self.fruit][0] * self.my_weights[3]) + (self.training_data[self.fruit][1] * self.my_weights[4]) + (self.training_data[self.fruit][2] * self.my_weights[5]) + self.my_biases[1]
            self.o1 = format(tf.sigmoid(self.o1), ".2f")
            self.o2 = format(tf.sigmoid(self.o2), ".2f")
            if self.o1 > self.o2:
                self.ai_guess = "apple"
            elif self.o1 < self.o2:
                self.ai_guess = "orange"
            else:
                self.ai_guess = "none"
            self.result_label = tkinter.Label(self.middle_bottom_frame, text=f"Output 1: {self.o1} Output 2: {self.o2} \nAi should guess: {self.training_data[self.fruit][3]} Ai guess is: {self.ai_guess}")
            self.result_label.pack(side="left")

    def menu(self):
        self.look.destroy()


class ViewYourAi:
    def __init__(self, master):
        try:
            input_file = open("ai.dat", 'rb')
            self.ai = pickle.load(input_file)
            input_file.close()
        except (FileNotFoundError, IOError, EOFError):
            self.ai = [[0, 0, 0, 0, 0, 0], [0, 0]]
        self.your_weights = self.ai[0]
        self.your_biases = self.ai[1]
        self.training_data = [[0.0, 0.0, 0.0, "orange"], [0.0, 0.0, 1.0, "orange"],
                              [0.0, 1.0, 0.0, "orange"], [1.0, 0.0, 1.0, "apple"], [1.0, 1.0, 0.0, "apple"], [1.0, 1.0, 1.0, "apple"]]
        self.fruit = 0
        self.o1 = (self.training_data[self.fruit][0] * self.your_weights[0]) + (self.training_data[self.fruit][1] * self.your_weights[1]) + (self.training_data[self.fruit][2] * self.your_weights[2]) + self.your_biases[0]
        self.o2 = (self.training_data[self.fruit][0] * self.your_weights[3]) + (self.training_data[self.fruit][1] * self.your_weights[4]) + (self.training_data[self.fruit][2] * self.your_weights[5]) + self.your_biases[1]
        self.o1 = format(tf.sigmoid(self.o1), ".2f")
        self.o2 = format(tf.sigmoid(self.o2), ".2f")
        if self.o1 > self.o2:
            self.ai_guess = "apple"
        elif self.o1 < self.o2:
            self.ai_guess = "orange"
        else:
            self.ai_guess = "none"

        # tkinter.Top_level() is like tkinter.Frame() but it opens in a new window
        self.look = tkinter.Toplevel(master)
        self.look.title('View my ai')

        # create Frames for this Top_level window
        self.top_frame = tkinter.Frame(self.look)
        self.middle_top_frame = tkinter.Frame(self.look)
        self.middle_bottom_frame = tkinter.Frame(self.look)
        self.bottom_frame = tkinter.Frame(self.look)

        # widgets for top frame - label and entry box for name
        self.inputs_text = tkinter.Label(self.top_frame, text=f'input 1 = {self.training_data[self.fruit][0]} input 2 = {self.training_data[self.fruit][2]} input 3 = {self.training_data[self.fruit][2]}')

        # pack top frame
        self.inputs_text.pack(side="left")

        # middle top frame label for weights and biases
        self.weight = tkinter.StringVar()
        self.weights_label = tkinter.Label(self.middle_top_frame, text=f"weights: {self.your_weights} biases: {self.your_biases}")

        # pack middle top frame
        self.weights_label.pack(side="left")

        # middle bottom frame - label for outputs
        self.outputs = tkinter.StringVar()
        self.result_label = tkinter.Label(self.middle_bottom_frame, text=f"Output 1: {self.o1} Output 2: {self.o2} \nAi should guess: {self.training_data[self.fruit][3]} Ai guess is: {self.ai_guess}")

        # pack middle bottom frame
        self.result_label.pack(side='left')

        # buttons for bottom frame
        self.menu_button = tkinter.Button(self.bottom_frame, text='Main Menu', command=self.menu)
        self.back_button = tkinter.Button(self.bottom_frame, text='back', command=self.back)
        self.next_button = tkinter.Button(self.bottom_frame, text='Next', command=self.next)

        # pack bottom frame
        self.menu_button.pack(side='left')
        self.back_button.pack(side='left')
        self.next_button.pack(side="left")

        # pack frames
        self.top_frame.pack()
        self.middle_top_frame.pack()
        self.middle_bottom_frame.pack()
        self.bottom_frame.pack()

    def next(self):
        if self.fruit <= 4:
            self.fruit += 1
            self.inputs_text.destroy()
            self.inputs_text = tkinter.Label(self.top_frame, text=f'input 1 = {self.training_data[self.fruit][0]} input 2 = {self.training_data[self.fruit][1]} input 3 = {self.training_data[self.fruit][2]}')
            self.inputs_text.pack(side="left")
            self.result_label.destroy()
            self.o1 = (self.training_data[self.fruit][0] * self.your_weights[0]) + (self.training_data[self.fruit][1] * self.your_weights[1]) + (self.training_data[self.fruit][2] * self.your_weights[2]) + self.your_biases[0]
            self.o2 = (self.training_data[self.fruit][0] * self.your_weights[3]) + (self.training_data[self.fruit][1] * self.your_weights[4]) + (self.training_data[self.fruit][2] * self.your_weights[5]) + self.your_biases[1]
            self.o1 = format(tf.sigmoid(self.o1), ".2f")
            self.o2 = format(tf.sigmoid(self.o2), ".2f")
            if self.o1 > self.o2:
                self.ai_guess = "apple"
            elif self.o1 < self.o2:
                self.ai_guess = "orange"
            else:
                self.ai_guess = "none"
            self.result_label = tkinter.Label(self.middle_bottom_frame, text=f"Output 1: {self.o1} Output 2: {self.o2} \nAi should guess: {self.training_data[self.fruit][3]} Ai guess is: {self.ai_guess}")
            self.result_label.pack(side="left")

    def back(self):
        if self.fruit >= 1:
            self.fruit -= 1
            self.inputs_text.destroy()
            self.inputs_text = tkinter.Label(self.top_frame, text=f'input 1 = {self.training_data[self.fruit][0]} input 2 = {self.training_data[self.fruit][1]} input 3 = {self.training_data[self.fruit][2]}')
            self.inputs_text.pack(side="left")
            self.result_label.destroy()
            self.o1 = (self.training_data[self.fruit][0] * self.your_weights[0]) + (self.training_data[self.fruit][1] * self.your_weights[1]) + (self.training_data[self.fruit][2] * self.your_weights[2]) + self.your_biases[0]
            self.o2 = (self.training_data[self.fruit][0] * self.your_weights[3]) + (self.training_data[self.fruit][1] * self.your_weights[4]) + (self.training_data[self.fruit][2] * self.your_weights[5]) + self.your_biases[1]
            self.o1 = format(tf.sigmoid(self.o1), ".2f")
            self.o2 = format(tf.sigmoid(self.o2), ".2f")
            if self.o1 > self.o2:
                self.ai_guess = "apple"
            elif self.o1 < self.o2:
                self.ai_guess = "orange"
            else:
                self.ai_guess = "none"
            self.result_label = tkinter.Label(self.middle_bottom_frame, text=f"Output 1: {self.o1} Output 2: {self.o2} \nAi should guess: {self.training_data[self.fruit][3]} Ai guess is: {self.ai_guess}")
            self.result_label.pack(side="left")

    def menu(self):
        self.look.destroy()

    def save(self):
        print("The data file has been updated with your changes.")
        save_file = open('customer_file.dat', 'wb')
        pickle.dump(self.ai, save_file)
        save_file.close()


class TrainAi:
    def __init__(self, master):
        try:
            input_file = open("ai.dat", 'rb')
            self.ai = pickle.load(input_file)
            input_file.close()
        except (FileNotFoundError, IOError, EOFError):
            self.ai = {}
        self.num_of_ai = 10000
        self.og_num = self.num_of_ai
        self.w = [0, 0, 0, 0, 0, 0]
        self.b = [0, 0]
        self.bf = 0
        self.bb = []
        self.bw = []
        self.training_data = [[0.0, 0.0, 0.0, "orange"], [0.0, 0.0, 1.0, "orange"], [0.0, 1.0, 0.0, "orange"], [1.0, 0.0, 1.0, "apple"], [1.0, 1.0, 0.0, "apple"], [1.0, 1.0, 1.0, "apple"]]
        self.r1 = 0
        self.num = 0
        self.fitness = 0
        self.loop1 = classes.Loop(self.fitness, self.w, self.b, self.training_data)
        self.fitness = self.loop1.get_fitness()

        # tkinter.Top_level() is like tkinter.Frame() but it opens in a new window
        self.train_1 = tkinter.Toplevel(master)
        self.train_1.title('Change customer')

        # create Frames for this Top_level window
        self.top_frame = tkinter.Frame(self.train_1)
        self.top_middle_frame = tkinter.Frame(self.train_1)
        self.middle_frame = tkinter.Frame(self.train_1)
        self.bottom_frame = tkinter.Frame(self.train_1)

        # widgets for top frame - label and entry box for name

        # pack top frame

        # middle frame - label for results
        self.value = ""
        self.result_label = tkinter.Label(self.middle_frame, text=f"Results: {self.value}")

        # pack Middle frame
        self.result_label.pack(side='left')

        # buttons for bottom frame
        self.menu_button = tkinter.Button(self.bottom_frame, text='Main Menu', command=self.back)
        self.till_done_button = tkinter.Button(self.bottom_frame, text='Train', command=self.train)
        self.save_button = tkinter.Button(self.bottom_frame, text='Save', command=self.save)

        # pack bottom frame
        self.menu_button.pack(side='left')
        self.till_done_button.pack(side="left")
        self.save_button.pack(side="left")

        # pack frames
        self.top_frame.pack()
        self.top_middle_frame.pack()
        self.middle_frame.pack()
        self.bottom_frame.pack()

    def back(self):
        self.train_1.destroy()

    def train(self):
        self.num_of_ai = self.og_num
        self.result_label.destroy()
        self.value = "Training"
        self.result_label = tkinter.Label(self.middle_frame, text=f"Results: {self.value}")
        self.result_label.pack(side='left')
        self.middle_frame.pack()
        while self.num_of_ai > 0:
            self.fitness = 0
            self.r1 = 0
            self.num = 0
            if self.num_of_ai != 10000:
                for _ in self.w:
                    self.w[self.num] = classes.format_random_weight()
                    self.num += 1

                self.num = 0
                for _ in self.b:
                    self.b[self.num] = classes.format_random_weight()
                    self.num += 1

                self.loop1 = classes.Loop(self.fitness, self.w, self.b, self.training_data)
                self.loop1.train(self.r1, self.fitness, self.w, self.b)
                self.fitness = self.loop1.get_fitness()
                self.w = self.loop1.get_w_copy()
                self.b = self.loop1.get_b_copy()

            print(self.fitness)
            print(self.w)
            print(self.b)

            if self.fitness > self.bf:
                self.bw = self.w
                self.bb = self.b
                self.bf = self.fitness

            self.num_of_ai -= 1
            if self.fitness == 6:
                self.num_of_ai = self.og_num - self.num_of_ai
                print("ai number", self.num_of_ai)
                self.num_of_ai = 0

        print("best fitness =", self.bf)
        print("best weights =", self.bw)
        print("best biases =", self.bb)
        self.result_label.destroy()
        self.value = "Training finished\nsave then go to view ai to see the ai"
        self.result_label = tkinter.Label(self.middle_frame, text=f"Results: {self.value}")
        self.result_label.pack(side='left')
        self.middle_frame.pack()

    def save(self):
        self.ai = [self.w, self.b]
        print("The ai has been saved.")
        save_file = open('ai.dat', 'wb')
        pickle.dump(self.ai, save_file)
        save_file.close()


def main():
    # create a window
    root = tkinter.Tk()
    # call the GUI and send it the root menu
    # use _ as variable name because the variable will not be needed after instantiating GUI
    # the GUI itself will handles the remaining program logic
    _ = CrudGUI(root)
    # control the mainloop from main instead of the class
    root.mainloop()


main()
