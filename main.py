import csv
import math
import random
import sys
try:
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
except:
    print("Modules: matplotlib, networkx and numpy are needed or visualization.\nAlgorithm will be run without visualization")

# Total population
population = 100
# placeholder for best result
best = None 
bestdist = float("inf")


'''
Read contents from csv file to create an matrix containing distance between locations
'''
def initialize(filename="map.csv"):

    matrix = []
    labels = []
    '''
    open file and read content line by line and column by colum
    add them to the matrix(list) to store the distances between locations
    check if value is an integer and not a location name if so add to the matrix
    '''
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=",")
        line = 0
        for row in csv_reader:
            if(line > 0):
                matrix.append([])
                for col in row:
                    try:
                        val = int(col)
                        matrix[line - 1].append(val)
                    except:
                        continue
            else:
                for col in row:
                    if len(col) > 0:
                        labels.append(col)
            line += 1

    return matrix, labels

'''
Create initial population given the initial map and population size
'''
def create_population(mat, pop=population):
    order = len(mat[0])
    init = [x for x in range(order)]
    generation = []
    for x in range(pop):
        temp = init.copy()
        random.shuffle(temp)
        generation.append(temp)
    return generation

'''
Caclulate fitnesses of the current generation
this calculates the fitness based on minimizing distances by using the inverse function
returns a list of fitness porbabilities which are calculated from the total fitness
'''
def fitness(gen):
    global best, bestdist
    fit = []
    dist = []
    total_fit = 0
    j = 0
    for index, x in enumerate(gen):
        tot = 0
        for ind, i in enumerate(x):
            if(ind < len(gen[0]) -1):
                tot += mat[i][x[ind+1]]
        dist.append(tot)
        fit.append(1 / tot)
        total_fit += (1/tot)
        if best is None or tot < bestdist:
            bestdist = tot
            best = gen[index].copy()
            j = index
    for x in range(len(fit)):
        fit[x] /= total_fit
    return fit, dist
    
'''
Choose an index randomly from fitness percentages
each index of the fitness function hold a probability calculated from the generations total fitness
using this method we choose a random fitness based upon probability of each generations member
this is done by subtracting each probability from a random number between 0 and 1 until the number is 0 or less
this grants members with higher probability a higher chance of being chosen
'''
def select(fit):
    index = 0
    rand = random.uniform(0, 1)

    while rand > 0:
        rand -= fit[index]
        index += 1

    index -= 1
    return index
    
'''
Crossover method
Implement crossover between to parents to create a child
swap a slice of parents a matrix into the child and get remaining information from parent b's order
'''
def crossover(a,b):
    l = len(a) - 1
    start = random.randint(0, l)
    end = random.randint(0, l) + 1
    c = a[start:end]
    for x in b:
        if x not in c:
            c.append(x)

    return c

'''
Mutate function
Swap two random indices to mutate child order
'''
def mutate(child):
    l = len(child) - 1
    a = random.randint(0,l)
    b = random.randint(0,l)
    temp = child[a]
    child[a] = child[b]
    child[b] = temp

'''
Calculate the distance between two points 
accepts two tuples containing an x and y coordinate 
form: (x,y)
'''
def calcdist(a,b):
    return math.sqrt( (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]) )

'''
Reads points from a file and builds adjacency matrix given the points coordinates
assumes all points are connected to one another but not themselves
'''
def buildMatrix(filename="points1.csv"):
    points = []
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=",")
        for row in csv_reader:
            points.append((int(row[0]), int(row[1])))
    matrix = []
    for i in range(len(points)):
        matrix.append([])
        for j in range(len(points)):
            if i == j:
                matrix[i].append(0)
            else:
                matrix[i].append(calcdist(points[i], points[j]))

    return matrix, points

    



if __name__ == "__main__":
    # 
    points_plot = None
    matrix_plot = None
    try:
        if sys.argv[1] == "points":
            mat, points = buildMatrix()
            points_plot = True
        elif sys.argv[1] == "matrix":
            # Create Map/Matrix of distances 
            mat, labels = initialize()
            matrix_plot = True
        else:
            raise Exception()
    except:
        print("Usage: main.py <argument>\npoints: passing points as an argument will run the algorithm on defined points from points.csv\nmatrix: passing matrix as an argument will run the algorithm on defined adjacency matrix from map.csv")
        exit()
    
    # Set initial values 
    iterations = 2000
    cur = 0
    prevdist = float("inf")
   

    # Create random Initial generation
    generation = create_population(mat)
    

    while(cur < iterations):

        newgen = []

        # Calculate fitness
        fit, dist = fitness(generation)

        for i in range(population):

            # Select parents from previous population 
            a = select(fit)
            b = select(fit)

            # Crossover
            child = crossover(generation[a],generation[b])
            
            # Mutation
            mutate(child)

            # Add child to next generation
            newgen.append(child.copy())

        # Print new best if a new best is found 
        if bestdist != prevdist:
            print("Best", best, bestdist)
        # increment iteration 
        cur += 1
        # Make new generation current generation
        generation = newgen.copy()
        prevdist = bestdist
    
    try:
        '''
        Draw visual representation 
        '''
        # Plot matrix path
        if matrix_plot:
            '''
            Adjacency Matrix plot
            '''
            matrix = np.array(mat)
            g = nx.from_numpy_matrix(matrix) 
            g = nx.relabel_nodes(g, dict(zip(range(len(labels)), labels)))
            g = nx.edge_subgraph(g, [(labels[best[i]], labels[best[i+1]]) for i in range(len(best) - 1)])
            pos = nx.shell_layout(g)
            '''
            Path 
            '''
            nx.draw_networkx(g)
            plt.title("Traveling Salesman Route")
            plt.show()
        # Plot points and connections
        elif points_plot:
            x = [points[best[i]][0] for i in range(len(points))]
            y = [points[best[i]][1] for i in range(len(points))]
            for i in range(len(best) - 1):
                plt.plot(x[i:i+2], y[i:i+2], 'ro-')
            plt.show()
    except:
        print("No visual output produced install necessary modules")

