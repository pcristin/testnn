#pylint: disable=no-member
import numpy as nmp
from scipy.special import expit

class neuralNetwork:
    
    #инициализация
    def __init__(self, inputnodes, hiddennodes, outputndoes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputndoes

        self.wih = nmp.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = nmp.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate

        self.activation_function = lambda x: expit(x)
        pass

    #тренировка нейронной сети
    def train(self, inputs_list, targets_list):
        inputs = nmp.array(inputs_list, ndmin=2).T
        targets = nmp.array(targets_list, ndmin=2).T

        hidden_inputs = nmp.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = nmp.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = nmp.dot(self.who.T, output_errors)
        
        self.who += self.lr * nmp.dot((output_errors * final_outputs * (1 - final_outputs)), nmp.transpose(hidden_outputs))
        self.wih += self.lr * nmp.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), nmp.transpose(inputs))
        pass

    #oпрос нейронной сети
    def query(self, inputs_list):
        inputs = nmp.array(inputs_list, ndmin=2).T
        
        hidden_inputs = nmp.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = nmp.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes,output_nodes, learning_rate)

training_data_file = open("mnist_dataset/mnist_test.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()
                #тренировка нейронной сети
epochs = 2

for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(",")
        inputs = (nmp.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
        targets = nmp.zeros(output_nodes) + 0.01

        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass


test_data_file = open("mnist_dataset/mnist_test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

                        #тесты для нейронной сети
scorecard = []
for record in test_data_list:
    all_values = record.split(",")
    correct_label = int(all_values[0])
    print(correct_label, " - истинный маркер")
    inputs = (nmp.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = nmp.argmax(outputs)
    print(label, " - ответ сети\n")
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard_array = nmp.asarray(scorecard)
print("Эффективность - ", scorecard_array.sum() / scorecard_array.size)