"""Adversarial attack MOAA.

We use the code from
https://github.com/phoenixwilliams/Black-Box-Sparse-Adversarial-Attack-via-Multi-Objective-Optimisation

"""
from copy import deepcopy
import math
import numpy as np
from operator import attrgetter
import torch
import time


class AttackMOAA:
    def __init__(self, params):
        self.params = params
        self.fitness = []
        self.data = []

    def completion_procedure(self, population, loss_function, fe, success):

        #print(success, fe)
        #print(1/0)

        adversarial_labels = []
        for soln in population.fronts[0]:
            adversarial_labels.append(loss_function.get_label(soln.generate_image()))

        d = {"front0_imgs": [soln.generate_image() for soln in population.fronts[0]],
             "queries": fe,
             "true_label": loss_function.true,
             "adversarial_labels": adversarial_labels,
             "front0_fitness": [soln.fitnesses for soln in population.fronts[0]],
             "fitness_process": self.fitness,
             "success": success
             }

        # print(d["true_label"], d["adversarial_labels"])
        # np.save(self.params["save_directory"], d, allow_pickle=True)

        return d

    def attack(self, loss_function):
        start = time.time()
        # print(loss_function(self.params["x"]))
        # print(self.params["n_pixels"])
        # Minimizes
        h, w, c = self.params["x"].shape[0:]
        pm = self.params["pm"]
        n_pixels = h * w
        all_pixels = np.arange(n_pixels)
        ones_prob = (1 - self.params["zero_probability"]) / 2
        init_solutions = [Solution(np.random.choice(all_pixels,
                                                    size=(self.params["eps"]), replace=False),
                                   np.random.choice([-1, 1, 0], size=(self.params["eps"], 3),
                                                    p=(ones_prob, ones_prob, self.params["zero_probability"])),
                                   self.params["x"].copy(), self.params["p_size"]) for _ in
                          range(self.params["pop_size"])]

        population = Population(init_solutions, loss_function, self.params["include_dist"])
        population.evaluate()
        fe = len(population.population)
        for it in range(1, self.params["iterations"]):
            #pm = p_selection(it, self.params["pm"], self.params["iterations"])
            pm = self.params["pm"]
            population.fronts = fast_nondominated_sort(population.population)

            adv_solns = population.find_adv_solns(self.params["max_dist"])
            if len(adv_solns) > 0:
                self.fitness.append(min(population.population, key=attrgetter('loss')).fitnesses)
                return self.completion_procedure(population, loss_function, fe, True)

            self.fitness.append(min(population.population, key=attrgetter('loss')).fitnesses)

            #print(fe, self.fitness[-1])

            for front in population.fronts:
                calculate_crowding_distance(front)
            parents = tournament_selection(population.population, self.params["tournament_size"])
            children = generate_offspring(parents,
                                          self.params["pc"],
                                          pm,
                                          all_pixels,
                                          self.params["zero_probability"])

            offsprings = Population(children, loss_function, self.params["include_dist"])
            fe += len(offsprings.population)
            offsprings.evaluate()
            population.population.extend(offsprings.population)
            population.fronts = fast_nondominated_sort(population.population)
            front_num = 0
            new_solutions = []
            while len(new_solutions) + len(population.fronts[front_num]) <= self.params["pop_size"]:
                calculate_crowding_distance(population.fronts[front_num])
                new_solutions.extend(population.fronts[front_num])
                front_num += 1

            calculate_crowding_distance(population.fronts[front_num])
            population.fronts[front_num].sort(key=attrgetter("crowding_distance"), reverse=True)
            new_solutions.extend(population.fronts[front_num][0:self.params["pop_size"] - len(new_solutions)])

            population = Population(new_solutions, loss_function, self.params["include_dist"])

        population.fronts = fast_nondominated_sort(population.population)
        self.fitness.append(min(population.population, key=attrgetter('loss')).fitnesses)
        return self.completion_procedure(population, loss_function, fe, False)


class Population:
    def __init__(self, solutions: list, loss_function, include_dist):
        self.population = solutions
        self.fronts = None
        self.loss_function = loss_function
        self.include_dist = include_dist

    def evaluate(self):
        for pi in self.population:
            pi.evaluate(self.loss_function, self.include_dist)

    def find_adv_solns(self, max_dist):
        adv_solns = []
        for pi in self.population:
            if pi.is_adversarial and pi.fitnesses[1] <= max_dist:
                adv_solns.append(pi)

        return adv_solns


class Solution:
    def __init__(self, pixels, values, x, p_size):
        self.pixels = pixels  # list of Integers
        self.values = values  # list of Binary tuples, i.e. [0, 1, 1]
        self.x = x  # (w x w x 3)
        self.fitnesses = []
        self.is_adversarial = None
        self.w = x.shape[0]
        self.delta = len(self.pixels)
        self.domination_count = None
        self.dominated_solutions = None
        self.rank = None
        self.crowding_distance = None

        self.loss = None
        self.p_size = p_size

    def copy(self):
        return deepcopy(self)

    def euc_distance(self, img):
        return np.sum((img - self.x.copy()) ** 2)

    def generate_image(self):
        x_adv = self.x.copy()
        for i in range(self.delta):
            x_adv[self.pixels[i] // self.w, self.pixels[i] % self.w] += (self.values[i] * self.p_size)

        return np.clip(x_adv, 0, 1)

    def evaluate(self, loss_function, include_dist):
        img_adv = self.generate_image()
        fs = loss_function(img_adv)
        self.is_adversarial = fs[0]  # Assume first element is boolean always
        self.fitnesses = fs[1:]
        if include_dist:
            dist = self.euc_distance(img_adv)
            self.fitnesses.append(dist)
        else:
            self.fitnesses.append(0)

        self.fitnesses = np.array(self.fitnesses)
        self.loss = fs[1]

    def dominates(self, soln):
        if self.is_adversarial is True and soln.is_adversarial is False:
            return True

        if self.is_adversarial is False and soln.is_adversarial is True:
            return False

        if self.is_adversarial is True and soln.is_adversarial is True:
            return True if self.fitnesses[1] < soln.fitnesses[1] else False

        if self.is_adversarial is False and soln.is_adversarial is False:
            return True if self.fitnesses[0] < soln.fitnesses[0] else False


class Targeted:
    def __init__(self, model, true, target, unormalize=False, to_pytorch=False):
        self.model = model
        self.true = true
        self.target = target
        self.unormalize = unormalize
        self.to_pytorch = to_pytorch

    def get_label(self, img):
        if self.unormalize:
            img_ = img * 255.

        else:
            img_ = img

        if self.to_pytorch:
            img_ = to_pytorch(img_)
            img_ = img_[None, :]
            preds = self.model.predict(img_).flatten()
            y = int(torch.argmax(preds))
        else:
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

        return y

    def __call__(self, img):

        if self.unormalize:
            img_ = img * 255.

        else:
            img_ = img

        if self.to_pytorch:
            img_ = to_pytorch(img_)
            img_ = img_[None, :]
            preds = self.model.predict(img_).flatten()
            y = int(torch.argmax(preds))
            preds = preds.tolist()
        else:
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

        is_adversarial = True if y == self.target else False
        #print("current label %d target label %d" % (y, self.target))
        f_target = preds[self.target]
        #preds[self.true] = -math.inf

        f_other = math.log(sum(math.exp(pi) for pi in preds))
        return [is_adversarial, f_other - f_target]


class UnTargeted:
    def __init__(self, model, true, unormalize=False, to_pytorch=False):
        self.model = model
        self.true = true
        self.unormalize = unormalize
        self.to_pytorch = to_pytorch

    def get_label(self, img):
        if self.unormalize:
            img_ = img * 255.

        else:
            img_ = img

        if self.to_pytorch:
            img_ = to_pytorch(img_)
            img_ = img_[None, :]
            preds = self.model.predict(img_).flatten()
            y = int(torch.argmax(preds))
        else:
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

        return y

    def __call__(self, img):
        if self.unormalize:
            img_ = img * 255.
        else:
            img_ = img

        if self.to_pytorch:
            img_ = to_pytorch(img_)
            img_ = img_[None, :]
            preds = self.model.predict(img_).flatten()
            y = int(torch.argmax(preds))
            preds = preds.tolist()
        else:
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

        is_adversarial = True if y != self.true else False

        f_true = math.log(math.exp(preds[self.true]) + 1e-30)
        preds[self.true] = -math.inf

        f_other = math.log(math.exp(max(preds)) + 1e-30)
        return [is_adversarial, float(f_true - f_other)]


def p_selection(it, p_init, n_queries):
    it = int(it / n_queries * 10000)
    if 0 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 5
    elif 500 < it <= 1000:
        p = p_init / 6
    elif 1000 < it <= 2000:
        p = p_init / 8
    elif 2000 < it <= 4000:
        p = p_init / 10
    elif 4000 < it <= 6000:
        p = p_init / 12
    elif 6000 < it <= 8000:
        p = p_init / 15
    elif 8000 < it:
        p = p_init / 20
    else:
        p = p_init

    return p


def to_pytorch(tensor_image):
    return torch.from_numpy(tensor_image).permute(2, 0, 1)


def fast_nondominated_sort(population):
    fronts = [[]]
    for individual in population:
        individual.domination_count = 0
        individual.dominated_solutions = []
        for other_individual in population:
            if individual.dominates(other_individual):
                individual.dominated_solutions.append(other_individual)
            elif other_individual.dominates(individual):
                individual.domination_count += 1
        if individual.domination_count == 0:
            individual.rank = 0
            fronts[0].append(individual)
    i = 0
    while len(fronts[i]) > 0:
        temp = []
        for individual in fronts[i]:
            for other_individual in individual.dominated_solutions:
                other_individual.domination_count -= 1
                if other_individual.domination_count == 0:
                    other_individual.rank = i + 1
                    temp.append(other_individual)
        i = i + 1
        fronts.append(temp)

    return fronts


def calculate_crowding_distance(front):
    if len(front) > 0:
        solutions_num = len(front)
        for individual in front:
            individual.crowding_distance = 0

        for m in range(len(front[0].fitnesses)):
            front.sort(key=lambda individual: individual.fitnesses[m])
            front[0].crowding_distance = 10 ** 9
            front[solutions_num - 1].crowding_distance = 10 ** 9
            m_values = [individual.fitnesses[m] for individual in front]
            scale = max(m_values) - min(m_values)
            if scale == 0: scale = 1
            for i in range(1, solutions_num - 1):
                front[i].crowding_distance += (front[i + 1].fitnesses[m] - front[i - 1].fitnesses[m]) / scale


def crowding_operator(individual, other_individual):
    if (individual.rank < other_individual.rank) or ((individual.rank == other_individual.rank) and (
            individual.crowding_distance > other_individual.crowding_distance)):
        return 1
    else:
        return -1


def __tournament(population, tournament_size):
    participants = np.random.choice(population, size=(tournament_size,), replace=False)
    best = None
    for participant in participants:
        if best is None or (
                crowding_operator(participant, best) == 1):  # and self.__choose_with_prob(self.tournament_prob)):
            best = participant

    return best


def tournament_selection(population, tournament_size):
    parents = []
    while len(parents) < len(population) // 2:
        parent1 = __tournament(population, tournament_size)
        parent2 = __tournament(population, tournament_size)

        parents.append([parent1, parent2])
    return parents


def mutation(soln, pm, all_pixels, zero_prob):
    all_pixels = all_pixels.copy()
    pixels = soln.pixels.copy()
    rgbs = soln.values.copy()

    eps_it = max([int(len(soln.pixels) * pm), 1])
    eps = len(soln.pixels)

    # select pixels to keep
    A_ = np.random.choice(eps, size=(eps - eps_it,), replace=False)
    new_pixels = pixels[A_]
    new_rgbs = rgbs[A_]

    # select new pixels to replace
    u_m = np.delete(all_pixels, pixels)
    B = np.random.choice(u_m, size=(eps_it,), replace=False)

    ones_prob = (1 - zero_prob) / 2
    rgbs_ = np.random.choice([-1, 1, 0], size=(eps_it, 3), p=(ones_prob, ones_prob, zero_prob))
    pixels_ = all_pixels[B]

    new_pixels = np.concatenate([new_pixels, pixels_], axis=0)
    new_rgbs = np.concatenate([new_rgbs, rgbs_], axis=0)

    soln.pixels = new_pixels
    soln.values = new_rgbs


def crossover(soln1, soln2, pc):
    l = max([int(len(soln1.pixels) * pc), 1])
    k = len(soln1.pixels)
    # S1 crossover with S2
    # 1. Generate set of different pixels in S2
    delta = np.asarray([pi for pi in range(k) if soln2.pixels[pi] not in soln1.pixels])

    offspring1 = soln1.copy()
    if len(delta)>0:
        l = l if l <= len(delta) else len(delta)
        switched_pixels = np.random.choice(delta, size=(l,))
        offspring1.pixels[switched_pixels] = soln2.pixels[switched_pixels].copy()
        offspring1.values[switched_pixels] = soln2.values[switched_pixels].copy()

    # S2 crossover with S1
    # 1. Generate set of different pixels in S2
    delta = np.asarray([pi for pi in range(k) if soln1.pixels[pi] not in soln2.pixels])
    offspring2 = soln1.copy()
    if len(delta)>0:
        l = l if l <= len(delta) else len(delta)
        switched_pixels = np.random.choice(delta, size=(l,))
        offspring2.pixels[switched_pixels] = soln1.pixels[switched_pixels].copy()
        offspring2.values[switched_pixels] = soln1.values[switched_pixels].copy()

    return offspring1, offspring2


def generate_offspring(parents, pc, pm, all_pixels, zero_prob):
    children = []
    for pi in parents:
        offspring1, offspring2 = crossover(pi[0], pi[1], pc)
        mutation(offspring1, pm, all_pixels, zero_prob)
        mutation(offspring2, pm, all_pixels, zero_prob)

        assert len(np.unique(offspring1.pixels)) == len(offspring1.pixels)
        assert len(np.unique(offspring2.pixels)) == len(offspring2.pixels)
        children.extend([offspring1, offspring2])

    return children