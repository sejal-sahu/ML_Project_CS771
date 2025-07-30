import numpy as np

def my_fit( words, verbose = False ):
	dt = Tree( min_leaf_size = 1, max_depth = 15 )
	dt.fit( words)
	return dt


class Tree:
	def __init__( self, min_leaf_size, max_depth ):
		self.root = None
		self.words = None
		self.min_leaf_size = min_leaf_size
		self.max_depth = max_depth
	
	def fit( self, words):
		self.words = words
		self.root = Node( depth = 0, parent = None, flag = 0, count = 0)
		# The root is trained with all the words
		self.root.fit( all_words = self.words, my_words_idx = np.arange( len( self.words ) ), min_leaf_size = self.min_leaf_size, max_depth = self.max_depth, isRoot = True)


class Node:
	# A node stores its own depth (root = depth 0), a link to its parent
	# A link to all the words as well as the words that reached that node
	# A dictionary is used to store the children of a non-leaf node.
	# Each child is paired with the response that selects that child.
	# A node also stores the query-response history that led to that node
	# Note: my_words_idx only stores indices and not the words themselves
	def __init__( self, depth, parent, flag, count ):
		self.depth = depth
		self.parent = parent
		self.flag = flag
		self.all_words = None
		self.my_words_idx = None
		self.children = {}
		self.is_leaf = True
		self.query_idx = None
		self.max_dict = []
		self.max_ind = []
		self.count = count
	
	# Each node must implement a get_query method that generates the
	# query that gets asked when we reach that node. Note that leaf nodes
	# also generate a query which is usually the final answer
	def get_query( self ):
		return self.query_idx
	
	# Each non-leaf node must implement a get_child method that takes a
	# response and selects one of the children based on that response
	def get_child( self, response ):
		# This case should not arise if things are working properly
		# Cannot return a child if I am a leaf so return myself as a default action
		if self.is_leaf:
			print( "Why is a leaf node being asked to produce a child? Melbot should look into this!!" )
			child = self
		else:
			# This should ideally not happen. The node should ensure that all possibilities
			# are covered, e.g. by having a catch-all response. Fix the model if this happens
			# For now, hack things by modifying the response to one that exists in the dictionary
			if response not in self.children:
				print( f"Unknown response {response} -- need to fix the model" )
				response = list(self.children.keys())[0]
			
			child = self.children[ response ]
			
		return child
	
	# Dummy leaf action -- just return the first word
	def process_leaf( self, my_words_idx):
		return my_words_idx[0]
	
	def reveal( self, word, query ):
		# Find out the intersections between the query and the word
		mask = [ *( '_' * len( word ) ) ]
		
		for i in range( min( len( word ), len( query ) ) ):
			if word[i] == query[i]:
				mask[i] = word[i]
		
		return ' '.join( mask )
	
	# Dummy node splitting action -- use a random word as query
	# Note that any word in the dictionary can be the query
	def process_node( self, all_words, my_words_idx, isRoot, count):
		# For the root we do not ask any query -- Melbot simply gives us the length of the secret word
		if isRoot == True:
			query_idx = -1
			query = ""
			split_dict = {}

			for idx in my_words_idx:
				mask = self.reveal(all_words[idx], query)
				if mask not in split_dict:
					split_dict[mask] = []

				split_dict[mask].append(idx)

			# if len( split_dict.items() ) < 2 and verbose:
				# print( "Warning: did not make any meaningful split with this query!" )

			return (query_idx, split_dict)
		# elif len(my_words_idx) == 0:
		# 	return (0, my_words_idx[0])
		# elif len(my_words_idx) < 0 and (self.flag == 2 | 3) and len(self.parent.max_ind) != 0:
		# 	self.max_dict = self.parent.max_dict
		# 	self.max_ind = self.parent.max_ind
		# 	return (self.max_ind[self.flag-2] , self.max_dict[self.flag-2]) 
		# elif len(my_words_idx) < 0 and self.flag == 1: 
		# 	max_query_idx = -1
		# 	max_query_split_dict = {}
		# 	max_entropy = -1
		# 	# query_idx = random.sample(range(0, len(all_words)), 4*K)
		# 	# query_idx1 = query_idx[0:K]
		# 	query_idx1 = np.random.randint( 0, len( my_words_idx ), size = min(30,len( my_words_idx )))
		# 	K = min(30,len( my_words_idx ))/3
		# 	query_idx = [my_words_idx[val] for val in query_idx1]
		# 	query = [all_words[val] for val in query_idx]
		# 	for i in range(len(query)):
		# 		split_dict1 = {}
		# 		for idx in my_words_idx:
		# 			mask = self.reveal(all_words[idx], query[i])
		# 			if mask not in split_dict1:
		# 				split_dict1[mask] = []
		# 			split_dict1[mask].append(idx)
		# 		# split_dict1 = self.split(my_words_idx,query[i])

		# 		for mask in split_dict1:
		# 			query_idx2 = query_idx[K+1:2*K]
		# 			query1 = [all_words[val] for val in query_idx2]
		# 			for j in range(len(query1)):
		# 				my_words_idx1 = split_dict1[mask]
		# 				split_dict2 = {}
		# 				for idx in my_words_idx1:
		# 					mask1 = self.reveal(all_words[idx], query1[j])
		# 					if mask1 not in split_dict2:
		# 						split_dict2[mask1] = []
		# 					split_dict2[mask1].append(idx)
		# 				# split_dict2 = self.split(my_words_idx1,query1[j])


		# 				for mask1 in split_dict2:
		# 					query_idx3 = query_idx[2*K+1:3*K]
		# 					query2 = [all_words[val] for val in query_idx2]
		# 					for k in range(len(query2)):
		# 						my_words_idx2 = split_dict2[mask1]
		# 						split_dict3 = {}
		# 						for idx in my_words_idx2:
		# 							mask2 = self.reveal(all_words[idx], query2[k])
		# 							if mask2 not in split_dict3:
		# 								split_dict3[mask2] = []
		# 							split_dict3[mask2].append(idx)
		# 						# split_dict3 = self.split(my_words_idx2,query2[k])
									
		# 						if max_entropy == -1:
		# 							max_entropy = len(split_dict3)
		# 							max_query_split_dict = split_dict1
		# 							max_query_idx = query_idx[i]
		# 							self.max_dict.append(split_dict2)
		# 							self.max_ind.append(query_idx[j])
		# 							self.max_dict.append(split_dict3)
		# 							self.max_ind.append (query_idx[k])
						
		# 						elif max_entropy < len(split_dict3):
		# 							max_entropy = len(split_dict3)
		# 							max_query_split_dict = split_dict1
		# 							max_query_idx = query_idx[i]
		# 							self.max_dict[0] = split_dict2
		# 							self.max_ind[0] = query_idx[j]
		# 							self.max_dict[1] = split_dict3
		# 							self.max_ind[1] = query_idx[k]
		# 			# if len( split_dict.items() ) < 2 and verbose:
		# 			# 	print( "Warning: did not make any meaningful split with this query!" )	
		# 	return ( max_query_idx, max_query_split_dict)
		else:
			K = 5
			max_query_idx = -1
			max_query_split_dict = {}
			max_size = -1 
			query_idx1 = np.random.randint( 0, len(my_words_idx), size = min(K,len(my_words_idx)) )
			query_idx = [my_words_idx[val] for val in query_idx1]
			query = [all_words[ val ] for val in query_idx]
   
			for i in range(len(query)):
				split_dict = {}
				for idx in my_words_idx:
					mask = self.reveal( all_words[ idx ], query[i] )
					if mask not in split_dict:
						split_dict[ mask ] = []
					
					split_dict[ mask ].append( idx )

				if max_size == -1:
					max_size = len(split_dict)
					max_query_split_dict = split_dict
					max_query_idx = query_idx[i]
          
				elif max_size < len(split_dict):
					max_size = len(split_dict)
					max_query_split_dict = split_dict
					max_query_idx = query_idx[i]
				# if len( split_dict.items() ) < 2 and verbose:
				# 	print( "Warning: did not make any meaningful split with this query!" )	
			return ( max_query_idx, max_query_split_dict)
	
	def fit( self, all_words, my_words_idx, min_leaf_size, max_depth, isRoot):
		self.all_words = all_words
		self.my_words_idx = my_words_idx
		
		# If the node is too small or too deep, make it a leaf
		# In general, can also include purity considerations into account
		if len( my_words_idx ) <= min_leaf_size or self.depth >= max_depth:
			self.is_leaf = True
			self.query_idx = self.process_leaf( self.my_words_idx)
		else:
			self.is_leaf = False
			( self.query_idx, split_dict ) = self.process_node( self.all_words, self.my_words_idx, isRoot, self.count)
			
			for ( i, ( response, split ) ) in enumerate( split_dict.items() ):
				flag1 = 0
				if self.flag == 0:
					flag1 = 1
				else:
					flag1 = self.flag%3 + 1 
				self.children[ response ] = Node( depth = self.depth + 1, parent = self, flag = flag1, count = self.count+1 )
				self.children[ response ].fit( self.all_words, split, min_leaf_size, max_depth, False)
