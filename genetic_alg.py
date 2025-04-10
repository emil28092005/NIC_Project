import torch
import torch.nn as nn
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, inputSize):
        super().__init__()
        self.hidden1 = nn.Linear(inputSize, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return torch.argmax(x)

class GeneticAlgorithm:
    def __init__(self, populationSize: int, mutationRate: float, percentageBest: float, percentageNew: float, inputSize: int = 5):
        self.populationSize = populationSize
        self.mutationRate = mutationRate
        self.percentageBest = percentageBest
        self.percentageNew = percentageNew
        self.inputSize = inputSize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.initialize_population()

    def create_ind(self):
        return NeuralNetwork(self.inputSize).to(self.device)

    def initialize_population(self):
        self.population = [self.create_ind() for _ in range(self.populationSize)]
    
    def crossover(self, parent1: NeuralNetwork, parent2: NeuralNetwork):
        child1 = self.create_ind()
        child2 = self.create_ind()
        point1 = len(child1.hidden1.weight.data) // 2
        point2 = len(child1.hidden2.weight.data) // 2
        child1.hidden1.weight.data = torch.cat((parent1.hidden1.weight.data[:point1], parent2.hidden1.weight.data[point1:]), dim=0)
        child2.hidden1.weight.data = torch.cat((parent2.hidden1.weight.data[:point1], parent1.hidden1.weight.data[point1:]), dim=0)
        child1.hidden2.weight.data = torch.cat((parent1.hidden2.weight.data[:point2], parent2.hidden2.weight.data[point2:]), dim=0)
        child2.hidden2.weight.data = torch.cat((parent2.hidden2.weight.data[:point2], parent1.hidden2.weight.data[point2:]), dim=0)
        child1.output.weight.data = parent1.output.weight.data.clone().detach()
        child2.output.weight.data = parent2.output.weight.data.clone().detach()
        return child1, child2

    # Mutation operator: Random mutation
    def mutate(self, model: NeuralNetwork):
        for param in model.parameters():
            if torch.rand(1).item() < self.mutationRate:
                param.data += torch.randn_like(param.data) * 0.1 * (1 if torch.rand(1).item() >= 0.5 else -1)
        return model

    def learn(self, fitness: list):
        self.population = [self.population[x] for x in np.argsort(fitness)[::-1]]
        numBest = int(self.populationSize * self.percentageBest)
        self.population = self.population[:numBest]
        while len(self.population) < self.populationSize - self.populationSize * self.percentageNew:
            parent1, parent2 = np.random.choice(self.population), np.random.choice(self.population)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            self.population.extend([child1, child2])
        while len(self.population) < self.populationSize: self.population.append(self.create_ind())
        while len(self.population) > self.populationSize: self.population.pop()
    
    def predict(self, data: list, i):
        data = torch.tensor(data, requires_grad=False).float().to(self.device)
        return self.population[i](data)
    