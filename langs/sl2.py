import numpy as np
import random
def _load_aux():
    from aux.generate_word import generate_words
    from aux.generate_geom import generate_dists
    return generate_words, generate_dists
import re

import matplotlib.pyplot as plt

from dataclasses import dataclass, fields
import os
from os import path
import json

@dataclass
class SL2_Config:
	N_FACT_MARK = 20 # Number of fact markers
	N_VOCAB = 75 # The size of the vocabulary
	N_GRAMMAR = 10 # Number of grammar tokens
	FACT_DENSITY = 0.04 # Frequency of facts
	RECALL_DENSITY = 0.04 # Frequency of facts needing to be recalled
	OTHER_DENSITY = 0.2 # Frequency of non-grammar words
	
	MC_CLOSE_TEMP = 0.2 # Temperature for the close MC of grammar tokens
	MC_FAR_TEMP = 0.8 # Temperature for the far MC of grammar tokens
	
	# These are parameters controlling the closeness of similar words
	# This is the temperature for the random walk on the geometry of non-grammar words
	RW_TEMP = 0.4

	USE_PRIOR = False # Determines if the prior should be used for unigram predictor
	MULTI_FACT = True # Determines if multiple facts can be associated with the same token
	USE_RECALL = True # Determins if fact matching should use fact# or recall# tokens
	NON_GRAM_TYPE = "rw"
	# OPTIONS
	# - rw    : random walk on the geometry given

	GRAMMAR_SIGMA = 2.5
	# NONGRAMMAR_SIGMA = 5.0
	# NONGRAMMAR_SIGMA = 2.5
	NONGRAMMAR_SIGMA = 0.9

	def __init__(self, **kwargs):
		for k, v in vars(type(self)).items():
			if not k.startswith("__") and not callable(v):
				setattr(self, k, v)

		for k, v in kwargs.items():
			setattr(self, k, v)

def stationary_distribution(P, tol=1e-12, max_iter=100_000):
    """Find the stationary distribution of P via power iteration.
 
    P : (n, n) row-stochastic transition matrix, P[a, b] = P(b | a)
    """
    n = P.shape[0]
    v = np.full(n, 1.0 / n)  # start uniform
    for _ in range(max_iter):
        v_next = v @ P
        v_next /= v_next.sum()  # guard against drift
        if np.max(np.abs(v_next - v)) < tol:
            v = v_next
            break
        v = v_next
    return v

class SL2:
	def __init__(self, config=None, load=False, dirname=None):
		# Prelude
		if load:
			self.load(dirname=dirname)

		else:
			assert config is not None
			self.config = config

			generate_words, _ = _load_aux()
			self.config.vocab = generate_words(self.config.N_VOCAB, self.config.N_GRAMMAR)
			self.config.vocab.sort(key = lambda x: (len(x), x)) # Sort by length, then alphabetically
		
			# We want to partition these words into a few groups: Grammar, and clustered synonyms
			self.config.grammar = self.config.vocab[:self.config.N_GRAMMAR]
			self.config.non_grammar = self.config.vocab[self.config.N_GRAMMAR:]
			self.config.non_grammar.sort()
			
			self.config.all_vocab = self.config.vocab + ["<start>"]
			self.config.all_vocab += ['fact%d' % i for i in range(1, self.config.N_FACT_MARK+1)]
			if self.config.USE_RECALL:
				self.config.all_vocab += ['recall%d' % i for i in range(1, self.config.N_FACT_MARK+1)]

		# Set up local pieces
		self.mc_keys = self.config.grammar + ['FACT', 'OTHER', 'RECALL']
		self.mc_keys2 = self.config.grammar + self.config.non_grammar + ['FACT', 'RECALL']

		self.vtoi = {v: i for i, v in enumerate(self.config.all_vocab)}
		self.mctoi = {v: i for i, v in enumerate(self.mc_keys)}
		self.mctoi2 = {v: i for i, v in enumerate(self.mc_keys2)}

		# If making, create the grammar and the geometry of non-grammar words, as well as the prior distribution over words
		if not load:
			self.make_grammar_geom()

			if self.config.USE_PRIOR:
				self.config.prior = np.random.normal(size=len(self.config.all_vocab))
				self.config.prior = np.exp(self.config.GRAMMAR_SIGMA * self.config.prior)
				self.config.prior /= self.config.prior.sum()
				self.config.prior = self.config.prior.tolist()

				for v in self.config.all_vocab:
					if re.fullmatch(r'(fact|recall)\d+', v):
						self.config.prior[self.vtoi[v]] = 1.0

				# Correct the non-grammar prior to be log-normal
				# This will iterate to find a V such that the stationary distribution of the non-grammar MC matches a log-normal distribution
				non_gram_target = np.random.normal(size=len(self.config.non_grammar))
				non_gram_target = np.exp(self.config.NONGRAMMAR_SIGMA * non_gram_target)
				non_gram_target /= non_gram_target.sum()

				non_gram_trans_array = np.zeros((len(self.config.non_grammar), len(self.config.non_grammar)))
				for i1, _ in enumerate(self.config.non_grammar):
					for i2, _ in enumerate(self.config.non_grammar):
						non_gram_trans_array[i1,i2] = np.exp(-self.dists[i1,i2] / self.config.RW_TEMP)
				non_gram_trans_array /= non_gram_trans_array.sum(axis=1, keepdims=True)

				non_gram_target = non_gram_target / non_gram_target.sum()
				log_non_gram_target = np.log(non_gram_target)

				# The distribution tilt
				V = np.zeros(non_gram_target.shape[0])

				for itr in range(1000):
					# Build P^(n) from current V: reweight columns by exp(V), renormalize rows
					W = non_gram_trans_array * np.exp(V)[None, :]          # broadcast over columns (states b)
					row_sums = W.sum(axis=1, keepdims=True)
					P = W / row_sums
			
					# Inner loop: stationary distribution of this P via power iteration
					pi_n = stationary_distribution(P, tol=1e-10, max_iter=1000)
			
					err = np.max(np.abs(pi_n - non_gram_target))
					if err < 1e-10:
						break
			
					# Outer update: nudge V toward matching the target marginal
					V = V + 0.5 * (log_non_gram_target - np.log(pi_n))
					V = V - V.max()  # recenter; only differences in V matter

				old_s = 0
				new_s = 0
				for i, v in enumerate(self.config.non_grammar):
					old_s += self.config.prior[self.vtoi[v]]
					# self.config.prior[self.vtoi[v]] *= np.exp(V[i])
					self.config.prior[self.vtoi[v]] = np.exp(V[i])
					new_s += self.config.prior[self.vtoi[v]]
				for i, v in enumerate(self.config.non_grammar):
					self.config.prior[self.vtoi[v]] *= old_s / new_s

			else:
				self.config.prior = [1/len(self.config.all_vocab)] * len(self.config.all_vocab)

		# Postlude
		self.mark_g = len(self.config.grammar)
		self.mark_ng = self.mark_g + len(self.config.non_grammar)
		if self.config.USE_RECALL:
			self.mark_f = - 2 * self.config.N_FACT_MARK
		else:
			self.mark_f = - self.config.N_FACT_MARK

		self.mc_grammar_np = {c1: {c2: np.array(self.mc_grammar[c1][c2]) for c2 in self.mc_keys2} for c1 in self.mc_keys2}


	def make_grammar_geom(self):
		# Get the geometry of non-grammar words
		_, generate_dists = _load_aux()
		self.dists = generate_dists(len(self.config.non_grammar))

		# We will want the grammar to follow the product of two 1st order MCs
		mc_near = {c1: {c2: np.exp(np.log(random.random()) / self.config.MC_CLOSE_TEMP) for c2 in self.mc_keys} for c1 in self.mc_keys2}
		mc_far = {c1: {c2: np.exp(np.log(random.random()) / self.config.MC_FAR_TEMP) for c2 in self.mc_keys} for c1 in self.mc_keys2}

		# We will want the grammar to follow a 2nd order MC
		self.mc_grammar = {c1: {c2: [mc_far[c1][c3] * mc_near[c2][c3] for c3 in self.mc_keys] for c2 in self.mc_keys2} for c1 in self.mc_keys2}

		# Sometimes the generated grammar is boring by repeating tokens over and over
		for c1 in self.mc_grammar.keys():
			for c2 in self.mc_grammar[c1].keys():
				if c2 in self.mc_keys:
					self.mc_grammar[c1][c2][self.mctoi[c2]] *= 0.0001

		# For saving the entropies of the MCs
		self.entropies = {c1: {c2: 0 for c2 in self.mc_keys2} for c1 in self.mc_keys2}
		
		# Normalize
		self.config.reserved = self.config.FACT_DENSITY + self.config.RECALL_DENSITY + self.config.OTHER_DENSITY
		for c1 in self.mc_grammar.keys():
			for c2 in self.mc_grammar[c1].keys():
				s = sum(self.mc_grammar[c1][c2][self.mctoi[c3]] for c3 in self.config.grammar)
				for c3 in self.config.grammar:
					self.mc_grammar[c1][c2][self.mctoi[c3]] *= (1 - self.config.reserved) / s
				self.mc_grammar[c1][c2][self.mctoi["OTHER"]] = self.config.OTHER_DENSITY
				self.mc_grammar[c1][c2][self.mctoi["RECALL"]] = self.config.RECALL_DENSITY
				self.mc_grammar[c1][c2][self.mctoi["FACT"]] = self.config.FACT_DENSITY

				# Calculate entropies
				self.entropies[c1][c2] = -sum(self.mc_grammar[c1][c2][self.mctoi[c3]] / (1 - self.config.reserved) * np.log(self.mc_grammar[c1][c2][self.mctoi[c3]] / (1 - self.config.reserved)) for c3 in self.config.grammar)

		# Report the average perplexity of the grammar MCs
		perplexities = [np.exp(self.entropies[c1][c2]) for c1 in self.mc_grammar.keys() for c2 in self.mc_grammar[c1].keys()]
		avg_perplexity = sum(perplexities) / len(perplexities)
		print("Average perplexity of grammar MCs:", avg_perplexity)

		plt.hist(perplexities, bins=20)
		plt.xlabel("Perplexity")
		plt.ylabel("Frequency")
		plt.title("Distribution of Perplexities")
		os.makedirs("fig", exist_ok=True)
		plt.savefig("fig/perplexity_distribution.png")
		plt.close()

		# Make it so that other does not repeat right after itself
		# l = ["OTHER"]
		l = self.config.non_grammar
		for c1 in self.mc_grammar.keys():
			for token in l:
				if token in self.mc_keys:
					self.mc_grammar[c1][token][self.mctoi[token]] = 0
				s = sum(self.mc_grammar[c1][token][self.mctoi[c3]] for c3 in self.mc_keys)
				for c3 in self.mc_keys:
					self.mc_grammar[c1][token][self.mctoi[c3]] /= s
		
		"""
		for c1 in self.mc_grammar.keys():
			for c2 in self.mc_grammar[c1].keys():
				s = 0
				for c3 in self.mc_grammar[c1][c2].keys():
					s += self.mc_grammar[c1][c2][c3]
				assert 0.95 < s and s < 1.05
		"""
		
		
		
	# Now we can start generating sentences
	def generate_sentence(self, generate_length, get_probs=False):
		self.np_prior = np.array(self.config.prior)
		facts = {'fact%d' % i: [] for i in range(1, self.config.N_FACT_MARK+1)}
		# exact match, not startswith: a real-word vocab (sl-eng) contains the word "fact"
		recall_markers = {'recall%d' % i for i in range(1, self.config.N_FACT_MARK+1)}

		last_word_index = np.random.choice(len(self.config.non_grammar))
		probs = np.zeros((generate_length, len(self.config.all_vocab)))
	
		word_history = ["<start>", random.choice(self.config.grammar), random.choice(self.config.grammar)]
		for g in self.config.grammar:
			probs[0, self.vtoi["<start>"]] = 1
			probs[1, self.vtoi[g]] = 1./len(self.config.grammar)
			probs[2, self.vtoi[g]] = 1./len(self.config.grammar)
		
		sentence = ' '.join(word_history)
		
		index = 0
		while index < generate_length:
			probs[index] = 0
	
			# Generation Probabilities
			needs_norm = False

			last_mc_grammar = self.mc_grammar[word_history[-2]][word_history[-1]]

			# Grammar
			f = last_mc_grammar[self.mctoi["OTHER"]] + last_mc_grammar[self.mctoi["FACT"]] + last_mc_grammar[self.mctoi["RECALL"]]
			if self.config.USE_PRIOR:
				probs[index, :self.mark_g] = last_mc_grammar[:self.mark_g] * self.np_prior[:self.mark_g]
				probs[index, :self.mark_g] *= (1-f) / probs[index, :self.mark_g].sum()
			else:
				probs[index, :self.mark_g] = last_mc_grammar[:self.mark_g]
				probs[index, :self.mark_g] *= (1-f) / probs[index, :self.mark_g].sum()

			# Non-Grammar
			if self.config.NON_GRAM_TYPE == "rw":
				sm = np.exp(-self.dists[last_word_index] / self.config.RW_TEMP)
				sm /= np.sum(sm)

			f = last_mc_grammar[self.mctoi["OTHER"]]
			if self.config.USE_PRIOR:
				probs[index, self.mark_g:self.mark_ng] = sm * self.np_prior[self.mark_g:self.mark_ng]
				probs[index, self.mark_g:self.mark_ng] *= f / probs[index, self.mark_g:self.mark_ng].sum()
			else:
				probs[index, self.mark_g:self.mark_ng] = sm
				probs[index, self.mark_g:self.mark_ng] *= f / probs[index, self.mark_g:self.mark_ng].sum()

			# Facts	
			if self.config.MULTI_FACT:
				options = facts.keys()
			else:
				options = [k for k in facts.keys() if len(facts[k]) == 0]
			if len(options) == 0:
				needs_norm = True # We'll have a prob vector that does not sum to 1
			else:
				f = last_mc_grammar[self.mctoi["FACT"]]
				for o in options:
					probs[index, self.vtoi[o]] = f / len(options)
	
			# Recall
			options = [int(k[4:]) for k in facts.keys() if len(facts[k]) > 0]
			if self.config.USE_RECALL: options = ["recall%d" % i for i in options]
			else         : options = ["fact%d"   % i for i in options]

			if len(options) == 0:
				needs_norm = True
			else:
				f = last_mc_grammar[self.mctoi["RECALL"]]
				for o in options:
					probs[index, self.vtoi[o]] += f / len(options)

			if needs_norm:
				probs[index] /= probs[index].sum()
			if not (0.999 < probs[index].sum() and probs[index].sum() < 1.001):
				print("Probabilities do not sum to 1:", probs[index].sum())
				assert False

			# Pick the next word
			# r = random.random()
			# next_word = ""
			# for w in self.config.all_vocab:
			# 	r -= probs[index, self.vtoi[w]]
			# 	if r <= 0:
			# 		next_word = w
			# 		break
			next_word = np.random.choice(self.config.all_vocab, p=probs[index])
	
			if next_word == "":
				assert False, "Shouldn't be here"

			# Fill out word history
			if next_word in facts:
				if self.config.USE_RECALL:
					word_history.append("FACT")
				else:
					t1 = self.mc_grammar[word_history[-2]][word_history[-1]][self.mctoi["FACT"]]
					if (not self.config.MULTI_FACT) and len(facts[next_word]) > 0:
						t1 = 0.
					t2 = self.mc_grammar[word_history[-2]][word_history[-1]][self.mctoi["RECALL"]]
					if len(facts[next_word]) == 0:
						t2 = 0.
					p = t1 / (t1 + t2)
					if random.random() < p:
						word_history.append("FACT")
					else:
						word_history.append("RECALL")

			elif next_word in recall_markers:
				word_history.append("RECALL")
			elif next_word in self.config.non_grammar:
				# word_history.append("OTHER")
				last_word_index = self.config.non_grammar.index(next_word)
				word_history.append(next_word)
			else:
				word_history.append(next_word)
			
			# Update things as needed
			if word_history[-1] == "FACT":
				sentence += " " + next_word
				index += 1
				if index >= generate_length:
					break

				fact_num = int(re.sub(r'\D','',next_word))
				next_word = random.choice(self.config.non_grammar)

				facts["fact%d" % fact_num].append(next_word)

				# reset the row: at index 0/1 it still holds the uniform-grammar placeholder
				probs[index] = 0
				for v in self.config.non_grammar:
					probs[index, self.vtoi[v]] = 1./len(self.config.non_grammar)

			elif word_history[-1] == 'RECALL':
				sentence += " " + next_word
				index += 1
				if index >= generate_length:
					break

				fact_num = int(re.sub(r'\D','',next_word))
				options = facts["fact%d" % fact_num]
				next_word = random.choice(options)

				probs[index] = 0   # reset, as in the fact branch
				for o in options:
					probs[index, self.vtoi[o]] = 1./len(options)			
				
			sentence += " " + next_word
			index += 1

		if get_probs:
			return ' '.join(sentence.split(' ')[:generate_length]), probs
		else:
			return ' '.join(sentence.split(' ')[:generate_length])


	def generate_many_sentences(self, n, length, print_status=False):
		sentences = []
		if print_status:
			for i in range(10):
				sentences += [self.generate_sentence(length) for _ in range(n//10)]
				print("Generated %d/%d sentences" % (len(sentences), n))
		else:
			sentences = [self.generate_sentence(length) for _ in range(n)]
		return sentences


	def get_mc_as_np(self):
		n = len(self.mc_keys)
		wtoi = {w: i for i, w in enumerate(self.mc_keys)}
		n2 = len(self.mc_keys2)
		wtoi2 = {w: i for i, w in enumerate(self.mc_keys2)}
		mc_np = np.zeros((n2, n2, n))
		for i1, c1 in enumerate(self.mc_keys2):
			for i2, c2 in enumerate(self.mc_keys2):
				for i3, c3 in enumerate(self.mc_keys):
					mc_np[i1,i2,i3] = self.mc_grammar[c1][c2][i3]
	
		return self.mc_keys, mc_np


	def get_geom_as_np(self):
		return self.config.grammar, self.dists


	def save(self, dirname=None):
		
		_, mc_np = self.get_mc_as_np()
		_, geom_np = self.get_geom_as_np()

		if not dirname:
			dirname = "sl2"
		
		pwd = os.getcwd()
		if not path.isdir(path.join(pwd, dirname)):
			os.mkdir(path.join(pwd, dirname))

		np.save(path.join(pwd, dirname, "mc.npy"), mc_np)
		np.save(path.join(pwd, dirname, "geom.npy"), geom_np)
		with open(path.join(pwd, dirname, "config.json"), "w") as outfile:
			json.dump(vars(self.config), outfile, indent=4)


	def load(self, dirname=None):
		if not dirname:
			dirname = "sl2"

		pwd = os.getcwd()
		assert path.isdir(path.join(pwd, dirname)), "Language specifications not set"

		with open(path.join(pwd, dirname, "config.json")) as infile:
			data = json.load(infile)
	
		known_fields = {f.name for f in fields(SL2_Config)}
		known_data = {k: v for k, v in data.items() if k in known_fields}
		extra_data = {k: v for k, v in data.items() if k not in known_fields}

		self.config = SL2_Config(**known_data)

		# bolt on anything that wasn't a declared field
		for k, v in extra_data.items():
			setattr(self.config, k, v)

		mc_np = np.load(path.join(pwd, dirname, "mc.npy"))
		geom_np = np.load(path.join(pwd, dirname, "geom.npy"))

		self.dists = geom_np

		self.mc_keys = self.config.grammar + ['FACT', 'OTHER', 'RECALL']
		self.mc_keys2 = self.config.grammar + self.config.non_grammar + ['FACT', 'RECALL']

		self.mc_grammar = {c1: {c2: [0 for _ in self.mc_keys] for c2 in self.mc_keys2} for c1 in self.mc_keys2}
		for i1, c1 in enumerate(self.mc_keys2):
			for i2, c2 in enumerate(self.mc_keys2):
				for i3, _ in enumerate(self.mc_keys):
					self.mc_grammar[c1][c2][i3] = mc_np[i1,i2,i3]


if __name__ == "__main__":
	# mode = "load" # in ["make", "load"]
	mode = "make" # in ["make", "load"]

	if mode == "make":
		config = SL2_Config()
		config.USE_RECALL = False
		config.MULTI_FACT = False
		sl = SL2(config=config)
	if mode == "load":
		sl = SL2(load=True)
		
	sentences = sl.generate_many_sentences(n=5, length=100)
	# Here's the completed sentence!
	print("SENTENCE:", sentences[4])
	
	if mode == "make":
		sl.save()
