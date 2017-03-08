# Percolation script

import numpy as np

GRID_SIZE = 10
PERC_CONST =  0.592746
# Apparently a triangle grid has a constant of exactly 1/2

def generateFixedCountGrid(gridLen = GRID_SIZE):
	nCells = gridLen * gridLen
	randPerm = np.random.permutation(nCells).reshape((gridLen, gridLen))
	grid = (randPerm < nCells * PERC_CONST).astype(int)
	return grid

def generateFixedProbGrid(gridLen = GRID_SIZE):
	nCells = gridLen * gridLen
	randGrid = (np.random.rand(nCells) < PERC_CONST).astype(int).reshape((gridLen, gridLen))
	return randGrid

print generateFixedCountGrid()
print generateFixedProbGrid()
