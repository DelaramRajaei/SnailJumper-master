import copy
import random
import numpy as np
from player import Player


def myFunc(e):
    return e.fitness


def top_k(players):
    players.sort(key=myFunc)
    return players


def tournament(players, n, q_size):
    new_population = []
    for i in range(n):
        candidates = random.sample(players, q_size)
        new_population.append(max(candidates, key=lambda x: x.fitness))
    return new_population


def calculate_probabilities(players):
    # total_fit = 0
    # for i in range (len(players)):
    #     total_fit += players.fitness
    total_fit = sum([x.fitness for x in players])
    relative_fitness = [x.fitness / total_fit for x in players]
    probabilities = [sum(relative_fitness[:i + 1]) for i in range(len(relative_fitness))]
    return probabilities


def roulette_wheel(players, n):
    new_population = []
    probabilities = calculate_probabilities(players)
    for n in range(n):
        r = random.random()
        for (i, individual) in enumerate(players):
            if r <= probabilities[i]:
                new_population.append(individual)
                break
    return new_population


def crossover_biases(c1, c2, b1, b2):
    if len(b1[0]) == 1 and len(b2[0]) == 1:
        c1[0] = np.zeros((1, len(c1)))
        c2[0] = np.zeros((1, len(c2)))
    elif len(b1[0]) == 1:
        c1[0] = b2
        c2[0] = b2
    elif len(b2[0]) == 1:
        c1[0] = b1
        c2[0] = b1
    else:
        c1[0] = np.append(b1[0:int(len(c1) / 2)], b2[int(len(c1) / 2):int(len(c1))])
        c2[0] = np.append(b2[0:int(len(c2) / 2)], b1[int(len(c2) / 2):int(len(c2))])
    return c1, c2


def write_file(players, n):
    average = sum([x.fitness for x in players]) / n
    players.sort(key=lambda x: x.fitness)
    max = players[n-1].fitness
    min = players[0].fitness
    f = open("generation_analysis.txt", "a")
    f.write(str(min) + " " + str(average) + " " + str(max) + "\n")


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        write_file(players, num_players)
        op = 2
        if op == 1:
            # Top-k algorithm
            players = top_k(players)
        elif op == 2:
            # Tournament
            players = tournament(players, num_players, 3)
        else:
            # Roulette wheel
            players = roulette_wheel(players, num_players)
        return players[: num_players]

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            op = 3
            if op == 1:
                new_players = self.generate(prev_players, num_players)
            elif op == 2:
                new_players = self.generate(roulette_wheel(prev_players, num_players), num_players)
            else:
                new_players = self.generate(tournament(prev_players, num_players, 3), num_players)
            return new_players

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

    def generate(self, players, n):
        new_players = []
        fitness = [x.fitness for x in players]
        for i in range(n):
            parents = random.choices(players, weights=fitness, k=2)
            childA, childB = self.crossover(self.clone_player(parents[0]), self.clone_player(parents[1]))
            childA = self.mutation(childA)
            childB = self.mutation(childB)
            new_players.append(childA)
            new_players.append(childB)
        return new_players

    def crossover(self, parentA, parentB):
        w11 = parentA.nn.weights[0]
        w12 = parentA.nn.weights[1]
        w21 = parentB.nn.weights[0]
        w22 = parentB.nn.weights[1]

        b11 = parentA.nn.biases[0]
        b12 = parentA.nn.biases[1]
        b21 = parentB.nn.biases[0]
        b22 = parentB.nn.biases[1]

        childA = self.clone_player(parentA)
        childB = self.clone_player(parentB)

        # Crossover weights
        for i in range(len(w11)):
            childA.nn.weights[0][i] = np.append(w11[i][0:int(len(w11) / 2)], w21[i][int(len(w11) / 2):len(w11)])
            childB.nn.weights[0][i] = np.append(w21[i][0:int(len(w11) / 2)], w11[i][int(len(w11) / 2):len(w11)])

        for i in range(len(w12)):
            childA.nn.weights[1][i] = np.append(w12[0:int(len(w12) / 2)], w22[i][int(len(w12) / 2):len(w12)])
            childB.nn.weights[1][i] = np.append(w22[i][0:int(len(w12) / 2)], w12[int(len(w12) / 2):len(w12)])

        # Crossover biases
        size_b_1 = len(childA.nn.biases[0])
        size_b_2 = len(childA.nn.biases[1])

        childA.nn.biases[0], childB.nn.biases[0] = crossover_biases(childA.nn.biases[0], childB.nn.biases[0], b11, b21)
        childA.nn.biases[1], childB.nn.biases[1] = crossover_biases(childA.nn.biases[1], childB.nn.biases[1], b12, b22)

        # childA.nn.biases[0] = np.append(b11[0:int(size_b_1 / 2)], b21[int(size_b_1 / 2):int(size_b_1)])
        # childB.nn.biases[0] = np.append(b21[0:int(size_b_1 / 2)], b11[int(size_b_1 / 2):int(size_b_1)])
        #
        # childA.nn.biases[1] = np.append(b12[0:int(size_b_2 / 2)], b22[int(size_b_2 / 2):int(size_b_2)])
        # childB.nn.biases[1] = np.append(b22[0:int(size_b_2 / 2)], b12[int(size_b_2 / 2):int(size_b_2)])

        return childA, childB

    def mutation(self, child):
        mutation_probability = 0.25
        if random.uniform(0, 1) < mutation_probability:
            child.nn.weights[0] += np.random.randn(child.nn.hidden_layer, child.nn.input_layer)
        if random.uniform(0, 1) < mutation_probability:
            child.nn.weights[1] += np.random.randn(child.nn.output_layer, child.nn.hidden_layer)
        if random.uniform(0, 1) < mutation_probability:
            child.nn.biases[0] = np.add(child.nn.biases[0], np.random.randn(1, child.nn.hidden_layer)[0])
        if random.uniform(0, 1) < mutation_probability:
            child.nn.biases[1] += np.random.randn(1, child.nn.output_layer)[0]
        return child
