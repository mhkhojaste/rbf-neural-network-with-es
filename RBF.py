import random
from deap import base
from deap import creator
from deap import tools
import array
import pandas as pd
import numpy as np
import math
from deap import algorithms
from sklearn.model_selection import train_test_split

np.seterr(over='ignore')

# read the data and create X and Y
data = pd.read_csv("New folder/regdata1500.csv")
X = data.drop(['D'], axis=1).values
Y = data['D'].values

# define parameters
X_array, test_x, Y_array, test_y = train_test_split(X, Y, test_size=0.4)
pop_num = 10 + (X_array.shape[0] // 5)
gen_nums = 7
number_of_generation = 1000
number_of_new_children = pop_num // 5
size_of_x = 3
MIN_STRATEGY = -1


# create gens
def generate_es(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind


# evaluate the individual using rbf neural network
def evaluate(individual):
    my_in = np.array(individual)
    my_in = my_in.reshape([gen_nums, size_of_x + 1])
    r = my_in[:, 0]
    r = r.reshape(r.shape[0], )
    v = my_in[:, 1:]
    G = np.zeros(shape=(X_array.shape[0], v.shape[0]))
    for i in range(v.shape[0]):
        G[:, i] = np.exp(-r[i] * (np.sum(np.subtract(X_array, v[i]) ** 2, axis=1) ** 0.5))
    if np.isnan(np.linalg.det(np.dot(np.transpose(G), G))) or np.linalg.det(np.dot(np.transpose(G), G)) == 0:
        return math.inf,
    if math.isnan(
            np.linalg.det(np.dot(np.transpose(G), G))) or math.isinf(np.linalg.det(np.dot(np.transpose(G), G))):
        return math.inf,

    W = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(G), G)), np.transpose(G)), Y_array)
    individual.my_w = W
    y_hat = np.dot(G, W)
    individual.y_hat = y_hat
    error = 0.5 * np.dot(np.transpose(np.subtract(y_hat, Y_array)), np.subtract(y_hat, Y_array))

    return round(error, 5),


def check_strategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children

        return wrappper

    return decorator


# use methods
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, my_w=None, y_hat=None,
               strategy=None)
creator.create("Strategy", array.array, typecode="d")

toolbox = base.Toolbox()
toolbox.register("individual", generate_es, creator.Individual, creator.Strategy,
                 gen_nums * (size_of_x + 1), 0, 1, -1, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

toolbox.decorate("mate", check_strategy(MIN_STRATEGY))
toolbox.decorate("mutate", check_strategy(MIN_STRATEGY))


def main():
    random.seed()
    MU, LAMBDA = 10, 100
    pop = toolbox.population(n=pop_num)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                              cxpb=0.6, mutpb=0.3, ngen=number_of_generation, stats=stats,
                                              halloffame=hof)

    return pop


if __name__ == "__main__":
    popp = main()
    best_ind = tools.selBest(popp, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    print(best_ind.my_w)
    print(np.round(best_ind.y_hat))
