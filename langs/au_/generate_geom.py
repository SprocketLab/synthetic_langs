import numpy as np
import random

class TreeNode:
	def __init__(self):
		self.val = -1 # Anything else means leaf
		self.left = None
		self.left_dist = 0
		self.left_vals = []
		self.right = None
		self.right_dist = 0
		self.right_vals = []

	def calc_dists(self, dists):
		if self.val != -1: return # A leaf

		self.left.calc_dists(dists)
		self.right.calc_dists(dists)

		for c1 in self.left_vals:
			dists[c1,c1] += self.left_dist

		for c1 in self.right_vals:
			dists[c1,c1] += self.right_dist

		for c1 in self.left_vals:
			for c2 in self.right_vals:
				dists[c1,c2] = dists[c1,c1] + dists[c2,c2]
				dists[c2,c1] = dists[c1,c1] + dists[c2,c2]

BASE = 2.5
# BASE = 1.5
def generate_tree(arr, depth=0):
	n = len(arr)
	assert n > 0
	
	if n == 1:
		node = TreeNode()
		node.val = arr[0]
		return node

	# Pick the splitting position
	# split = random.randint(0, (n-2) // 2) + random.randint(1, (n+1) // 2)
	split = random.randint(1, n-1)
	assert split > 0 and split < n

	node = TreeNode()
	node.left_vals = arr[:split]
	node.left = generate_tree(arr[:split], depth+1)
	# node.left_dist = (random.random() + 1) * BASE**(-depth)
	node.left_dist = (2*random.random() + 1) * BASE**(-depth)
	node.right_vals = arr[split:]
	node.right = generate_tree(arr[split:], depth+1)
	node.right_dist = (random.random() + 1) * BASE**(-depth)
	# node.right_dist = (2*random.random() + 1) * BASE**(-depth)
	return node

def generate_dists(n):
	tree = generate_tree(list(range(n)))
	dists = np.zeros((n, n))
	
	tree.calc_dists(dists)
	for i in range(n):
		dists[i, i] = 0 # Fix an artifact from calculation
	
	return dists


if __name__ == "__main__":
	n = 10
	dists = generate_dists(n)
	for i in range(n):
		for j in range(n):
			print("%4.2f " % dists[i][j], end="")
		print()
