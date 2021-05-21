import random
from math import sqrt

#Neuron - this is what processes the data and returns an answer
class Neuron:
    def __init__(self,numInputs,learningRate):
        self.lr = learningRate #The rate at which the weights are adjusted
        self.weights = [] #A coefficient for a feature - the goal is to determine the ideal weight for each feature
        for _ in range(numInputs): # a for loop which initialises the weights randomly to begin with
            self.weights.append(2*random.random()-1) # at the start it generates a random weight between -1 and 1

    #This function returns the sum of the inputs multiplied by their weights
    def getSum(self,inputs):
        sum = 0
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]
        
        return sum

    #This function "feeds" the neuron with data - and asks it to guess an answer
    def feed(self,inputs):
        return self.activate(self.getSum(inputs))

    #This returns an answer - 1 meaning the point is outside the circle, -1 meaning the point lies on or is within the circle
    def activate(self,sum):
        if sum > 0:
            return 1
        else:
            return -1
    
    #This gives the neuron an answer along with inputs, so it can learn from its mistakes and adjust weights accordingly
    def train(self,inputs,answer):
        guess = self.feed(inputs)
        error = answer - guess
        for i in range(len(self.weights)):
            self.weights[i] += self.lr * error * inputs[i]

#This holds a group of neurons
class Layer:
    def __init__(self,numNeurons,numInputs,learningRate):
        self.neurons = [] # The neurons in the list
        for _ in range(numNeurons): # loops numNeurons amount of times, creating a neuron object
            self.neurons.append(Neuron(numInputs,learningRate))

    #Goes through each neurons and feeds inputs
    def feed(self,inputs):
        for neuron in self.neurons:
            neuron.feed(inputs)
    
    #Trains each neuron
    def train(self,inputs,answer):
        for neuron in self.neurons:
            neuron.train(inputs,answer)

    #Returns list of all neurons
    def getNeurons(self):
        return self.neurons

class Network:
    def __init__(self,numInput,learningRate,numInputNeurons=3,numHiddenNeurons=3,numOutputNeurons=1):
        self.inputLayer = Layer(numInputNeurons,numInput,learningRate) #This is the first layer which receives raw input and outputs a sum
        self.hiddenLayer = Layer(numHiddenNeurons,numInput,learningRate) #This receives the sum of the input layer, and outputs its own sum
        self.outputLayer = Layer(numOutputNeurons,numInput,learningRate) #THis receieves the sum of the hidden layer, and outputs an answer
        self.inCircle = 0 # The number of points which lie in the circle
        self.totalInputs = 0 # The number of points which have been used

    def feed(self,inputs):
        self.inputLayer.feed(inputs) # feeds raw inputs to input layer getting a sum
        self.backpropagate(inputs,self.inputLayer,self.hiddenLayer,False) # feeds the output of the input layer to hidden layer
        return self.backpropagate(inputs,self.hiddenLayer,self.outputLayer,False) # feeds output of hidden layer and returns the sum of the output layer

    def train(self,inputs,answer):
        self.inputLayer.train(inputs,answer) # trains the input layer neurons
        self.backpropagate(inputs,self.inputLayer,self.hiddenLayer,True,answer) # trains the hidden layer neurons using data from input neurons
        self.backpropagate(inputs,self.hiddenLayer,self.outputLayer,True,answer) # trains the output layer neuron using data from hidden neurons

    def backpropagate(self,inputs,layerFrom,layerTo,train,answer=None):
        for neuron in layerTo.getNeurons(): #for each neuron in the layer to pass data to
            newInputs = [] # create a list of new inputs
            for i in range(len(layerFrom.getNeurons())): # for each neuron in the layer to receive data from
                newInputs.append(layerFrom.getNeurons()[i].getSum(inputs)) # append the sum of the inputs from the layer to get data from to pass into the new layer
            if train: # if training the neuron
                return neuron.train(newInputs,answer) # provide an answer
            else: # if feeding the neuron
                return neuron.feed(newInputs) # return the guess of the network

    #This function takes in a series of inputs and collects data on whether or not the point is within the circle or not
    #This is run after the network is trained - so all answers here are purely based on the network itself
    def estimatePi(self,inputs):
        if self.feed(inputs) == -1: #if within the circle
            self.inCircle += 1 # increment the number of points in the circle

        self.totalInputs += 1 # this is incremented regardless of whether the point is within the circle or not

    def getIn(self):
        return self.inCircle # returns the number of points within the circle

    def getTotal(self):
        return self.totalInputs # returns the number of points used to estimate pi

class Trainer:
    def __init__(self,x,y,answer):
        self.inputs = [x,y,1] # takes a random x and y value, the last input being a "bias" which is always 1
        self.answer = answer # the answer which will be given to the neuron

    def getInputs(self):
        return self.inputs # return inputs

    def getAnswer(self):
        return self.answer # return answer

def f(x):
    return sqrt(1-x**2) # a simple function to draw a circle - derived from x^2 + y^2 = 1

def train(network,iterations):
    trainers = [] # a list of all the trainer objects which will train the network
    for _ in range(iterations): # loops iterations amount of times
        x = random.uniform(0,1) # generates a uniform number between 0 and 1 - the maximum value of a point on the circle will be (0,1)
        y = random.uniform(0,1) # generates a uniform number between 0 and 1 for the y value
        if f(x) < y: #passes the x value into the function above, and if the point is below the y value, this training point is outside the circle
            a = 1 # shows it is outside the circle
        else:
            a = -1 # shows it is within the circle
        trainers.append(Trainer(x,y,a)) # adds the trainer to the list

    for trainer in trainers: # for each trainer
        network.train(trainer.getInputs(),trainer.getAnswer()) # train the network using the inputs and answer

def feed(network,iterations):
    for _ in range(iterations): # loops 'iterations' amount of times
        x = random.uniform(0,1) # generates a random x and y value to test
        y = random.uniform(0,1)

        network.feed([x,y,1]) # feeds the network - and the network will guess an answer based on training

def pi(network,iterations):
    for _ in range(iterations):
        x = random.uniform(0,1) # generates a random x and y value to estimate pi with
        y = random.uniform(0,1)

        network.estimatePi([x,y,1])

iterations = 100000 # number of iterations
n = Network(3,0.01) # creates a network object with neurons taking 3 inputs, and all having a learning rate of 0.01
train(n,iterations) # train the network iterations amount of times
pi(n,iterations) # estimate pi with iterations number of values
print(4*n.getIn()/n.getTotal()) # print the estimated value of pi