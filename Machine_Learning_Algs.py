import heapq
import numpy as np
from random import randint

trainfeature_file = open("YOUR_TRAINING_X_HERE")
trainlabel_file = open("YOUR_TEST_X_HERE")
valfeature_file = open("YOUR_TRAIN_Y_HERE)
vallabel_file = open("YOUR_TEST_Y_HERE")

############################################################
                       #K_NEAREST_NEIGHBORS#
############################################################

#Parses and readies file for training
def file_split(feature_file, label_file=None):
	data = []
	if label_file!= None:
		label =[]
		for line in label_file:
			label.append(int(line[0]))
		count=0
	for line in feature_file:
		data_line = map(float, line.split(","))
		if label_file!=None:
			data_line.append(label[count])
			count+=1
		data.append(data_line)
	return data

#Euclidean Distance of two data points
def distance(data1, data2):
	total = 0
	for i in range(len(data1)):
		total+= (data1[i]-data2[i])**2
	return total

#Implementation of KNN Algorithm
def knn(train_data, val_feature, k):
	knn_files = []
	for i in k:
		knn_file = open("./digit_classes_"+str(i)+".csv", "w")
		knn_files.append(knn_file)
	for i in range(len(val_feature)):
		dist_queue = PriorityQueue()
		for j in range(len(train_data)):
			d = distance(val_feature[i], train_data[j])
			dist_queue.push((train_data[j][784], d), d)
		k_neighbors = [[] for x in range(len(k))]
		for x in range(len(k)):
			q = k[x-1]
			if x-1 <0:
				q = 0
			for y in range(q, k[x]):
				value = dist_queue.pop()
				for z in range(x, len(k)):
					k_neighbors[z].append(value)
		for p in range(len(k)):
			classes = [[0,0] for x in range(10)]
			for (c, dis) in k_neighbors[p]:
				classes[c][0]+=1
				classes[c][1]+=dis
			max_val = (0,float("inf"))
			max_class = -1
			for c in range(len(classes)):
				if max_val[0]<classes[c][0]:
					max_val = classes[c]
					max_class = c
				elif max_val[0]==classes[c][0]:
					if max_val[1]>classes[c]:
						max_val = classes[c]
						max_class = c
			knn_files[p].write(str(max_class)+"\n")
	return

#PriorityQueue Implementation
class PriorityQueue:
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

#Script Execution for K=1,2,5,10,25
train = file_split(trainfeature_file, trainlabel_file)
val = file_split(valfeature_file)
knn(train, val, (1, 2, 5, 10, 25))

#####################################################################
                       #RANDOM_FOREST#
#####################################################################
                       
trainfeature_file = open("YOUR_TRAINING_X_HERE")
trainlabel_file = open("YOUR_TEST_X_HERE")
valfeature_file = open("YOUR_TRAIN_Y_HERE)
vallabel_file = open("YOUR_TEST_Y_HERE")
test_file = open("YOUR_TEST_FILE_HERE)

#Implementation of a Node Class
class Node:
 	def __init__(self, feature_index, threshold):
 		self.feature_i = feature_index
 		self.threshold = threshold
 		self.leftchild = None
 		self.rightchild = None
 		self.classtype = None

#Entropy Function of a data point
def entropy(cur_data, master_data):
	zeroratio = 0
	for i in cur_data:
		#check to see if gets y value
		if master_data[i][-1] == 0:
			zeroratio = zeroratio + 1
	zeroratio = float(zeroratio)/float(len(cur_data))
	if zeroratio == 0 or zeroratio == 1:
		return 0
	oneratio = 1 - zeroratio
	return -((zeroratio * np.log(zeroratio)) + (oneratio * np.log(oneratio)))

#Goodness measure for data points
def goodness(cur_data, left_data, right_data, master_data):
	H = entropy(cur_data, master_data)
	S = len(cur_data)
	SL = len(left_data)
	SR = len(right_data)
	HL = entropy(left_data, master_data)
	HR = entropy(right_data, master_data)
	return H - (((float(SL)/float(S)) * HL) + ((float(SR)/float(S)) * HR))

#Generates indices from 0 to max index
def gen_indices(length, max_index):
	result = [0] * length
	for i in range(0, length):
		randomnumber = randint(0, max_index)
		result[i] = randomnumber
	return result
                 
#Returns boolean if stop condition has been reached
def reachedStopC(train_data, ran_indices):
	if len(ran_indices) == 1:
		return True 
	firstindex = ran_indices[0]
	firstclass = train_data[firstindex][-1]
	for i in ran_indices:
		if firstclass != train_data[i][-1]:
			return False
	return True

#Implementation of building a RF tree
def buildTree(train_data, ran_indices):
	#base case
	if reachedStopC(train_data, ran_indices):
		leafnode = Node(None, None)
		leafnode.classtype = train_data[ran_indices[0]][-1]
		return leafnode

	#these are indices to access ran_indices
	feature_ranind = gen_indices(8, 56)
	bestfeature = None
	bestthreshold = None
	bestgoodness = float("-inf")
	bestleftgroup = None
	bestrightgroup = None
	for cur_feature in feature_ranind:
		sortedcolumn = sorted(set([train_data[x][cur_feature] for x in ran_indices]))
		possthresholds = sortedcolumn[:(len(sortedcolumn)-1)]
		if len(possthresholds)==0:
			leafnode = Node(None, None)
			mysum = 0
			for i in ran_indices:
				mysum += train_data[i][-1]
			mysum = float(mysum) / float(len(ran_indices))
			if mysum >= .50:
				leafnode.classtype = 1
			else:
				leafnode.classtype = 0
			return leafnode
		for cur_thresh in possthresholds:
			leftgroup = []
			rightgroup = []
			for data_index in ran_indices:
				if train_data[data_index][cur_feature] <= cur_thresh:
					leftgroup.append(data_index)
				else:
					rightgroup.append(data_index)
			cur_goodness = goodness(ran_indices, leftgroup, rightgroup, train_data)
			if (cur_goodness > bestgoodness):
				bestfeature = cur_feature
				bestthreshold = cur_thresh
				bestgoodness = cur_goodness
				bestleftgroup = leftgroup
				bestrightgroup = rightgroup
	currentnode = Node(bestfeature, bestthreshold)
	currentnode.leftchild = buildTree(train_data, bestleftgroup)
	currentnode.rightchild = buildTree(train_data, bestrightgroup)
	return currentnode

#Random Forest Tree Traversal and classification
def traverseTree(observation, treenode):
	if treenode.threshold == None:
		return treenode.classtype
	else:
		curfeature = treenode.feature_i
		curthresh = treenode.threshold
		if observation[curfeature] <= curthresh:
			return traverseTree(observation, treenode.leftchild)
		else:
			return traverseTree(observation, treenode.rightchild)

#Random Forest Classifier
def randomforest(train_data, val_feature, T=1):
	treelist = []
	predict_vect = [None] * len(val_feature)
	for i in range(T): 
		baggingsub = gen_indices(len(train_data), len(train_data) - 1)
		curr_tree = buildTree(train_data, baggingsub)
		treelist.append(curr_tree)
	for j in range(0, len(val_feature)):
		curr_observ = val_feature[j]
		class_array = [None] * len(treelist)
		for k in range(len(treelist)):
			curr_class = traverseTree(curr_observ, treelist[k])
			class_array[k] = curr_class
		curr_sum = float(sum(class_array))/float(len(treelist))
		if curr_sum > .50:
			predict_vect[j] = 1
		else:
			predict_vect[j] = 0
	return predict_vect

#Script Execution
train = file_split(trainfeature_file, trainlabel_file)
val = file_split(valfeature_file)
output_num = [1, 2, 5, 10, 25]
for num in output_num:
	rf_classifications = randomforest(train, val, num)
	labels = open("./emailOutput"+ str(num) + ".csv", "w")
	for label in rf_classifications:
		labels.write(str(label)+"\n")
test = file_split(test_file)
rf_classifications = randomforest(train, test, 10)
labels = open("./emailOutput.csv", "w")
for label in rf_classifications:
	labels.write(str(label)+"\n")