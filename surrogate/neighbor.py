from deap import base, creator, tools, algorithms
import numpy as np
from surrogate.distance import *

import _pickle as cPickle

def cPickle_clone(x):
    return cPickle.loads(cPickle.dumps(x))


def mutate(feature_values, indpb, toolbox, individual):
    new_individual = toolbox.clone(individual)
    for feature_idx in range(0, len(individual)):
        values = feature_values[feature_idx]
        if np.random.random() <= indpb:
            val = np.random.choice(values, 1)[0]
            new_individual[feature_idx] = val
    return new_individual,

def calculate_feature_values(train_data, columns, target_name, discrete_cols,size=1000):
    
    feature_names = [ col for col in list(columns) if col != target_name]
    feature_values = dict()
    
    for i, col in enumerate(feature_names):
        train_values = list(train_data[col])
        if col in discrete_cols:
            uniq_values, counts = np.unique(train_values, return_counts=True)
            prob = 1.0 * counts / np.sum(counts)
            new_values = np.random.choice(uniq_values, size=size, p=prob)
            new_values = np.concatenate((uniq_values, new_values), axis=0)
        else:
            mu = np.mean(train_values)
            sigma = np.std(train_values)
            new_values = np.random.normal(mu, sigma, size)
            new_values = np.concatenate((train_values, new_values), axis=0)

        feature_values[i] = new_values
    return feature_values

class GeneticAlgorithmPermutationGenerator(object):
    def __init__(
            self,
            idx_features,
            discrete,
            continuous,
            target_name,
            bb,
            scaler,
            alpha1 = 0.5,
            alpha2 = 0.5,
            eta1 = 1.0,
            eta2 = 0.0,
            tournsize = 3,
            cxpb = 0.5,
            mutpb = 0.2,
            ):
        self.bb = bb
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.eta1 = eta1
        self.eta2 = eta2
        self.tournsize = tournsize
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.idx_features = idx_features
        self.discrete = discrete
        self.continuous = continuous
        self.target_name = target_name
        self.scaler = scaler

    def record_init(self, x):
        return x

    def get_toolbox(self, x, feature_values, population_size, init_func, evaluate_func):
        creator.create("fitness", base.Fitness, weights=(1.0,))
        creator.create("individual", list, fitness=creator.fitness)
        
        toolbox = base.Toolbox()
        toolbox.register("feature_values", init_func, x)
        toolbox.register(
            "individual",
            tools.initIterate,
            creator.individual,
            toolbox.feature_values
            )
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual,
            n=population_size
            )
        
        toolbox.register("clone", cPickle_clone)
        toolbox.register("evaluate", evaluate_func, x)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutate, feature_values, self.mutpb, toolbox)
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)
        
        return toolbox

    def get_oversample(self, population, halloffame):
        fitness_values = [p.fitness.wvalues[0] for p in population]
        fitness_values = sorted(fitness_values)
        fitness_diff = [fitness_values[i+1] - fitness_values[i] for i in range(0, len(fitness_values)-1)]

        index = np.max(np.argwhere(fitness_diff == np.amax(fitness_diff)).flatten().tolist())
        fitness_value_thr = fitness_values[index]
        
        oversample = list()
        
        for p in population:
            if p.fitness.wvalues[0] > fitness_value_thr:
                oversample.append(list(p))
                
        for h in halloffame:
            if h.fitness.wvalues[0] > fitness_value_thr:
                oversample.append(list(h))
                
        return oversample 

    def fit(
            self,
            toolbox,
            population_size,
            ngen,
            hall_of_fame_ratio=0.1,
            verbose=False
            ):
    
        hall_of_fame_size = int(np.round(population_size * hall_of_fame_ratio))
        
        population = toolbox.population(n=population_size)
        hall_of_fame = tools.HallOfFame(hall_of_fame_size)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        population, logbook = algorithms.eaSimple(
            population,
            toolbox,
            cxpb=self.cxpb,
            mutpb=self.mutpb,
            ngen=ngen,
            stats=stats,
            halloffame=hall_of_fame,
            verbose=verbose
            )
        
        return population, hall_of_fame, logbook

    def generate_data(
            self,
            x,
            feature_values,
            n_gen = 10,
            population_size=1000,
            hall_of_fame_ratio=0.1,
            verbose=False,
        ):

        neighbors = []

        x = x.reshape(-1,)

        size_for_each = int(population_size / 2)

        eq_toolbox =  self.get_toolbox(x, feature_values, size_for_each, self.record_init, self.fitness_eq)

        eq_population, eq_hall_of_fame, eq_logbook = self.fit(eq_toolbox, size_for_each, n_gen, hall_of_fame_ratio, verbose)

        neighbors.extend(self.get_oversample(eq_population, eq_hall_of_fame))

        neq_toolbox = self.get_toolbox(x, feature_values, size_for_each, self.record_init, self.fitness_neq)

        neq_population, neq_hall_of_fame, neq_logbook = self.fit(neq_toolbox, size_for_each, n_gen, hall_of_fame_ratio, verbose)

        neighbors.extend(self.get_oversample(neq_population, neq_hall_of_fame))

        return neighbors

    def distance_function(self, x, y):
        xd = [x[att] for att in self.discrete if att != self.target_name]
        wd = 0.0
        dd = 0.0
        if len(xd) > 0:
            yd = [y[att] for att in self.discrete if att != self.target_name]
            wd = 1.0 * len(self.discrete) / (len(self.discrete) + len(self.continuous))
            dd = simple_match_distance(xd, yd)

        xc = np.array([x[att] for att in self.continuous])
        wc = 0.0
        cd = 0.0
        if len(xd) > 0:
            yc = np.array([y[att] for att in self.continuous])
            wc = 1.0 * len(self.continuous) / (len(self.discrete) + len(self.continuous))
            cd = normalized_euclidean_distance(xc, yc)

        return wd * dd + wc * cd


    def fitness_eq(self, x0, x1):
        ## Function to maximise
        x0d = {self.idx_features[i]: val for i, val in enumerate(x0)}
        x1d = {self.idx_features[i]: val for i, val in enumerate(x1)}
        sim_ratio = 1.0 - self.distance_function(x0d,x1d)
        record_similarity = 0.0 if sim_ratio >= self.eta1 else sim_ratio
    
        y0 = self.bb.predict(self.scaler.transform(np.asarray(x0).reshape(1, -1)))[0]
        y1 = self.bb.predict(self.scaler.transform(np.asarray(x1).reshape(1, -1)))[0]
        # target_similarity = 1.0 if y0 == y1 else 0.0
        target_similarity = -abs(y1-np.round(y0))
    
        fitness_value = self.alpha1 * record_similarity + self.alpha2 * target_similarity

        return fitness_value,


    def fitness_neq(self, x0, x1):
        # similar_different_outcome
        x0d = {self.idx_features[i]: val for i, val in enumerate(x0)}
        x1d = {self.idx_features[i]: val for i, val in enumerate(x1)}

        # zero if is too similar
        sim_ratio = 1.0 - self.distance_function(x0d,x1d)

        record_similarity = 0.0 if sim_ratio >= self.eta1 else sim_ratio

        y0 = self.bb.predict(self.scaler.transform(np.asarray(x0).reshape(1, -1)))[0]
        y1 = self.bb.predict(self.scaler.transform(np.asarray(x1).reshape(1, -1)))[0]
        # target_similarity = 1.0 if y0 != y1 else 0.0
        target_similarity = abs(y1 - np.round(y0))

        fitness_value = self.alpha1 * record_similarity + self.alpha2 * target_similarity

        return fitness_value,



