class GeneticAlgorithmPermutationGenerator(object):
    def __init__(
            self,
            distance_function,
            idx_features,
            discrete,
            continuous,
            target_name,
            bb,
            alpha1 = 0.5,
            alpha2 = 0.5,
            eta1 = 1.0,
            eta2 = 0.0,
            tournsize = 3,
            cxpb = 0.5,
            mutpb = 0.2,
            ngen = 10,
            return_logbook=False,
            ):
        self.bb = bb
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.eta1 = eta1
        self.eta2 = eta2
        self.tournsize = tournsize
        self.cxpb = cxpb
        self.mutpb = mutpb,
        self.ngen = ngen
        self.return_logbook = return_logbook
        self.idx_features = idx_features
        self.discrete = discrete
        self.continuous = continuous
        self.target_name = target_name


    def get_toolbox(x, feature_values, init_func, evaluate_func):
        creator.create("fitness", base.Fitness, weights=(1.0,))
        creator.create("individual", list, fitness=creator.fitness)
        
        toolbox = base.Toolbox()
        toolbox.register("feature_values", init_func, x)
        toolbox.register("individual", tools.initIterate, creator.individual, toolbox.feature_values)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=population_size)
        
        toolbox.register("clone", cPickle_clone)
        toolbox.register("evaluate", evaluate, record, bb, alpha1, alpha2, eta, discrete, continuous,
                        class_name, idx_features, distance_function)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutate, feature_values, mutpb, toolbox)
        toolbox.register("select", tools.selTournament, tournsize=tournsize)
        
        return toolbox


    def generate_data(self,):
        pass



    def mixed_distance(self, x, y):
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
        x0d = {self.idx_features[i]: val for i, val in enumerate(x0)}
        x1d = {self.idx_features[i]: val for i, val in enumerate(x1)}
        sim_ratio = 1.0 - self.distance_function(
                x0d,
                x1d,
                self.discrete,   
                self.continuous,
                self.target_name
                )
        record_similarity = 0.0 if sim_ratio >= eta else sim_ratio
    
        y0 = self.bb.predict(np.asarray(x0).reshape(1, -1))[0]
        y1 = self.bb.predict(np.asarray(x1).reshape(1, -1))[0]
        target_similarity = 1.0 if y0 == y1 else 0.0
    
        fitness_value = self.alpha1 * record_similarity + self.alpha2 * target_similarity

        return fitness_value


    def fitness_sdo(self, x0, x1):
        # similar_different_outcome
        x0d = {self.idx_features[i]: val for i, val in enumerate(x0)}
        x1d = {self.idx_features[i]: val for i, val in enumerate(x1)}

        # zero if is too similar
        sim_ratio = 1.0 - self.distance_function(
            x0d,
            x1d,
            self.discrete,
            self.continuous,
            self.target_name
            )

        record_similarity = 0.0 if sim_ratio >= eta else sim_ratio

        y0 = self.bb.predict(np.asarray(x0).reshape(1, -1))[0]
        y1 = self.bb.predict(np.asarray(x1).reshape(1, -1))[0]
        target_similarity = 1.0 if y0 != y1 else 0.0

        fitness_value = self.alpha1 * record_similarity + self.alpha2 * target_similarity
        return fitness_value


