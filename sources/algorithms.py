import random
import numpy as np


class PSO:
    def __init__(self, data):
        print("Initializing PSO...")
        self.data = data
        self.particles = []

    def clustering(self, num_particles, inertia=1, cognitive=2, social=2, version='global'):
        print("Performing clustering...")
        for i in range(num_particles):
            self.particles.append(self.initializeParticle())
        print(self.particles)

    def initializeParticle(self):
        return Particle()


class Particle:
    MIN_POS = 0
    MAX_POS = 100
    MIN_VEL = 5
    MAX_VEL = 50

    def __init__(self):
        self.position = random.randint(self.MIN_POS, self.MAX_POS)
        self.velocity = random.randint(self.MIN_VEL, self.MAX_VEL)
        self.best_pos = self.position
        self.best_val = np.inf

    def move(self):
        pass

    def calculate_fitness(self):
        pass
