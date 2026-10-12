import numpy as np
import random
import string

def generate_words(n, n_grammar=-1):
	chars = list(string.ascii_lowercase)
	chars.append(' ')

	# Add-1 smoothing: start with a value of 1 instead of 0
	# NOTE: This can be decreased later

	# mc = {c1: {c2: {c3: 1. for c3 in chars} for c2 in chars} for c1 in chars}
	mc = {c1: {c2: {c3: 0.1 for c3 in chars} for c2 in chars} for c1 in chars}

	# with open('words_alpha.txt') as infile:
	with open('aux/words_alpha.txt') as infile:
		data = infile.read().lower().split('\n')

	# Add data to MC

	for word in data:
		if len(word) >= 1:
			mc[' '][' '][word[0]] += 1
		if len(word) >= 2:
			mc[' '][word[0]][word[1]] += 1
		for i in range(2, len(word)):
			mc[word[0]][word[1]][word[2]] += 1
		if len(word) >= 3:
			mc[word[-2]][word[-1]][' '] += 1

	# Normalize the probabilities

	for c1 in mc.keys():
		for c2 in mc[c1].keys():
			s = sum([mc[c1][c2][c3] for c3 in mc[c1][c2].keys()])
			for c3 in mc[c1][c2].keys():
				mc[c1][c2][c3] /= s

	# Generate n words based on these probabilities
	words = []

	if n_grammar == -1:
		while len(words) < n:
			word = '  '
			while not (word[-1] == ' ' and word[-2] != ' ') and len(word) <= 20:
				r = random.random()
				for itr, c in enumerate(chars):
					r -= mc[word[-2]][word[-1]][c]
					if r <= 0:
						word += c
						break
	
			word = word[2:-1]

			if word == "ass": continue # This keeps happening
			if word in words: continue
	
			words.append(word) # Strip extra spaces

	else:
		# We will be trying to make words that are grammar and non_grammar
		while len(words) < n_grammar:
			word = '  '
			while not (word[-1] == ' ' and word[-2] != ' ') and len(word) <= 20:
				r = random.random()
				for itr, c in enumerate(chars):
					r -= mc[word[-2]][word[-1]][c]
					if r <= 0:
						word += c
						break
	
			word = word[2:-1]

			if len(word) > 3: continue # Word is too long for grammar
			if word == "ass": continue # This keeps happening
			if word in words: continue
	
			words.append(word) # Strip extra spaces
			
		# Now time to add the rest of the words
		while len(words) < n:
			word = '  '
			while not (word[-1] == ' ' and word[-2] != ' ') and len(word) <= 20:
				r = random.random()
				for itr, c in enumerate(chars):
					r -= mc[word[-2]][word[-1]][c]
					if r <= 0:
						word += c
						break
	
			word = word[2:-1]

			if len(word) < 5: continue # Word is too short for non-grammar
			if word in words: continue
	
			words.append(word) # Strip extra spaces

	return words

if __name__ == "__main__":
	print("Hello!")

	words = generate_words(5)
	print(words)
