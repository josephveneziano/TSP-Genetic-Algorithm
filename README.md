# TSP-Genetic-Algorithm
Traveling Salesman Problem solved used genetic algorithm written using python 3


## Usage

The file main.py is the main file to run the Traveling Salesman Problem using a genetic aglorithm. This program accepts two types of data sets including adjacency matrix and a list of points containing (x,y) values both from a csv file.

To run the algorithm:

    python main.py <argument>

Arguments:

    points: reads the file points.csv which contains a list of points containing x and y values for each point

    matrix: reads the file map.csv which contains an adjacency matrix with the distances between each location corresponding to row and column

## Needed Modules

* numpy
* matplotlib
* networkx

The following modules can all be install using the python package manager pip with their corrseponding names. If modules are not installed solution be written to console but no visual representation will be created.