# Percolation script

import numpy as np

GRID_SIZE = 100
PERC_CONST =  0.592746
# Apparently a triangle grid has a constant of exactly 1/2

class Grid:
	def __init__(self, n = GRID_SIZE, fixedCount = True):
		self.n = n
		if fixedCount:
			self.grid = self.fixedCountGrid(n)
		else:
			self.grid = self.fixedProbGrid(n)
		self.startNode = (-1,None)
		self.endNode = (n,None)
		self.hasPath = int(self.bfs())

	def bfs(self):
		queue = [self.startNode]
		visited = [self.startNode]
		while len(queue) > 0:
			node = queue.pop(0)
			if node == self.endNode:
				return True
			neighbors = self.cellNeighbors(node)
			for cell in neighbors:
				if cell not in visited:
					queue.append(cell)
					visited.append(cell)
		return False

	def cellNeighbors(self, (row, col)):
		n = self.n
		if self.startNode == (row,col):
			return [(0,i) for i in range(n) if self.isFilled(0,i)]
		elif self.endNode == (row,col):
			return [(n-1,i) for i in range(n) if self.isFilled(n-1,i)]

		neighbors = [(row, col+1), (row, col-1), (row+1, col), (row-1, col)]
		neighbors = [(r,c) for (r,c) in neighbors if \
					0 <= r < n and 0 <= c < n and self.isFilled(r,c)]
		if row == 0:
			neighbors.append(self.startNode)
		if row == n-1:
			neighbors.append(self.endNode)
		return neighbors

	def isFilled(self, row, col):
		return self.grid[row][col]

	def fixedCountGrid(self, n = GRID_SIZE):
		nCells = n * n
		randPerm = np.random.permutation(nCells).reshape((n, n))
		grid = (randPerm < nCells * PERC_CONST).astype(int)
		return grid

	def fixedProbGrid(self, n = GRID_SIZE):
		nCells = n * n
		randGrid = (np.random.rand(nCells) < PERC_CONST).astype(int).reshape((n, n))
		return randGrid


# count = 0
# for i in range(50):
# 	aoeu = Grid(GRID_SIZE, True)
# 	hasPath = aoeu.hasPath
# 	if hasPath:
# 		count+=1
# 	print hasPath
# print "Count:", count



