import random
import numpy as np
import torch


class Swarm:

    def __init__(self, mof, x, v, x_min, x_max,
                 inertia_weight=0.7, phi1=2., phi2=2., phi3=2.):
        self.mof = mof
        self.x = x
        self.v = v
        self.num_part = len(mof)
        self.fitness = np.zeros(self.num_part)

        self.x_min = x_min
        self.x_max = x_max
        self.inertia_weight = inertia_weight
        self.phi1 = phi1
        self.phi2 = phi2
        self.phi3 = phi3

        self.unscaled_scores = {}
        self.scaled_scores = {}
        self.desirability_scores = {}

        self.particle_best_fitness = self.fitness
        self.particle_best_x = x

        self.swarm_best_fitness = 0
        self.swarm_best_x = x[0]
        self.best_mof = mof[0]

        self.history_swarm_best_x = [x[0]]
        self.history_swarm_best_step_mof = []

    def next_step(self):
        u1 = torch.tensor(np.random.uniform(0, self.phi1, [self.num_part, 1]))
        u2 = torch.tensor(np.random.uniform(0, self.phi2, [self.num_part, 1]))
        u3 = torch.tensor(np.random.uniform(0, self.phi3, [self.num_part, 1]))
        v_u1 = u1 * (self.particle_best_x - self.x)
        v_u2 = u2 * (self.swarm_best_x - self.x)
        v_u3 = u3 * (random.choice(self.history_swarm_best_x) - self.x)
        self.v = self.inertia_weight * self.v + v_u1 + v_u2 + v_u3
        self.x += self.v
        self.x = torch.stack([torch.clamp(self.x[:, i], self.x_min[i], self.x_max[i]) for i in range(self.x.shape[1])],
                             dim=1)

    def update_fitness(self, fitness):
        self.fitness = fitness
        best_idx = np.argmax(fitness)
        best_fitness = fitness[best_idx]
        self.history_swarm_best_step_mof.append({**{'fitness': best_fitness}, **self.mof[best_idx]})
        if best_fitness > self.swarm_best_fitness:
            self.history_swarm_best_x.append(self.x[best_idx])
            self.swarm_best_fitness = np.copy(best_fitness)
            self.swarm_best_x = self.x[best_idx]
            self.best_mof = self.mof[best_idx]

        for index in range(self.num_part):
            if self.particle_best_fitness[index] < self.fitness[index]:
                self.particle_best_fitness[index] = self.fitness[index]
                self.particle_best_x[index] = self.x[index]

    def __repr__(self):
        return 'mso.swarm.Swarm num_part={} best_fitness={}'.format(self.num_part,
                                                                    self.swarm_best_fitness)

    @classmethod
    def from_query(cls, init_mof, init_emb, num_part, v_min=-0.6, v_max=0.6, *args, **kwargs):
        if isinstance(init_mof, list):
            idxs = np.random.randint(0, len(init_mof), size=num_part)
            mof = [init_mof[i] for i in idxs]
            x = init_emb[idxs]
        else:
            mof = init_mof
            mof = num_part * [mof]
            x = np.tile(init_emb, [num_part, 1])
        v = torch.tensor(np.random.uniform(v_min, v_max, [num_part, init_emb.shape[-1]]))
        swarm = Swarm(mof=mof, x=x.cpu(), v=v, *args, **kwargs)
        return swarm
