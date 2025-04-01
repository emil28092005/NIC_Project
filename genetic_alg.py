import torch
import torch.nn as nn
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, inputSize):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(inputSize, 10)
        self.output = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return torch.log_softmax(x)

class GeneticAlgorithm:
    def __init__(self, populationSize: int, mutationRate: float, percentageBest: float, inputSize: int = 5):
        self.populationSize = populationSize
        self.mutationRate = mutationRate
        self.percentageBest = percentageBest
        self.inputSize = inputSize
        self.initialize_population()

    def initialize_population(self):
        self.population = [NeuralNetwork(self.inputSize) for _ in range(self.populationSize)]
    
    def crossover(self, parent1: NeuralNetwork, parent2: NeuralNetwork):
        child1 = NeuralNetwork()
        child2 = NeuralNetwork()
        point = len(child1.hidden.weight.data) // 2
        print([x for x in child1.parameters()])
        child1.hidden.weight.data = torch.cat((parent1.hidden.weight.data[:point], parent2.hidden.weight.data[point:]), dim=0)
        child2.hidden.weight.data = torch.cat((parent2.hidden.weight.data[:point], parent1.hidden.weight.data[point:]), dim=0)
        child1.output.weight.data = parent1.output.weight.data.clone().detach()
        child2.output.weight.data = parent2.output.weight.data.clone().detach()
        return child1, child2

    # Mutation operator: Random mutation
    def mutate(self, model: NeuralNetwork):
        for param in model.parameters():
            if torch.rand(1).item() < self.mutationRate:
                param.data += torch.randn_like(param.data) * 0.1
        return model

    def learn(self, fitness: list):
        self.population = [self.population[x] for x in np.argsort(fitness)]
        numBest = int(self.populationSize * self.percentageBest)
        self.population = self.population[:numBest]
        while len(self.population) < self.populationSize:
            parent1, parent2 = np.random.choice(self.population), np.random.choice(self.population)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            self.population.extend([child1, child2])
        while len(self.population) > self.populationSize: self.population.pop()
    
    def predict(self, data: list):
        result = []
        for i in range(self.populationSize):
            result.append(self.population[i](data[i]))
        return result