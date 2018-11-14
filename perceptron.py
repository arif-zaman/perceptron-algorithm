# 1005031
# CSE474
# 21/9/2015

import copy
import random
import numpy as np
import matplotlib.pyplot as plt

Weight = []
WS = []
Train = None
Test = None

p = 0.7
hs = 0
feature = 0
classType = 0
sampleSize = 0


def trainData():
	global Test,Train,Weight
	global feature,classType,sampleSize

	with open("Train.txt","r+") as fhandle:
		temp = fhandle.readline().replace("\n","")
		temp2 = []
		for value in temp.split():
			temp2.append(int(value))

		feature = temp2[0]
		classType = temp2[1]
		sampleSize = temp2[2]
		#print feature,classType,sampleSize

		Weight = [ (random.randint(1,sampleSize) / float(sampleSize)) for x in xrange(feature)]
		Weight.append(-(1.0/classType))


		Test = [[ 0 for y in xrange(feature+1)] for x in xrange(sampleSize)]
		Train = [[ 0 for y in xrange(feature+1)] for x in xrange(sampleSize)]	

		#print Test
		#print Train
		#print Weight

		i = 0
		for line in fhandle:
			temp = line.replace("\n","")
			j = 0
			for value in temp.split():
				Train[i][j] = float(value)
				j += 1

			i += 1


	#print Train
	#print len(Train)


def testData():
	with open("Test.txt","r+") as fhandle:	
		i = 0
		for line in fhandle:
			temp = line.replace("\n","")
			j = 0
			for value in temp.split():
				Test[i][j] = float(value)
				j += 1

			i += 1


	#print Test
	#print len(Test)


def fileInput():
	trainData()
	testData()


def update(i,matrix):
	delX = 0
	cost = 0.0
	if matrix[i][-1] == 1:
		delX = -1
	else:
		delX = 1

	for j in xrange(feature):
		cost = cost + Weight[j]*matrix[i][j]*delX

	if cost >= 0:
		return True
	else:
		return False

	
def testModel(matrix):
	count = 0
	for i in xrange(len(Test)):
		cost = 0.0
		for j in xrange(feature):
			cost = cost+matrix[j]*Test[i][j] # sum(Wi * Xi)

		cost = cost+matrix[-1] # sum(Wi * Xi) + Wo

		if cost >= 0:
			#print "Class : ",1
			if Test[i][-1] == 1:
				count += 1
		else:
			#print "Class : ",2
			if Test[i][-1] == 2:
				count += 1

	return count


## Algorithms
def perceptron(iteration):
	global Weight,count
	count = 0
	temp = copy.deepcopy(Weight)
	while (True):
		Y = []
		count += 1
		for i in xrange(len(Train)):
			if update(i,Train):
				Y.append(i)

		#print "Y = ",len(Y)
		for i in Y:
			if Train[i][-1] == 1:
				Weight[0] = Weight[0] + p*Train[i][0]
				Weight[1] = Weight[1] + p*Train[i][1]

			else:
				Weight[0] = Weight[0] - p*Train[i][0]
				Weight[1] = Weight[1] - p*Train[i][1]

		if len(Y) == 0 or count >= iteration:
			break

	count = testModel(Weight)
	#print ("\tIteration : %3d \t Success : %2d \t Equation : a = %3.3f, b = %3.3f, c = %3.3f") % (iteration,count,Weight[0],Weight[1],Weight[2])
	w = copy.deepcopy(Weight)
	Weight = copy.deepcopy(temp)
	return count,w


def rewardPunishment(iteration):
	global Weight,count
	count = 0
	temp = copy.deepcopy(Weight)
	while (True):
		count += 1
		for i in xrange(len(Train)):
			delX = 0
			cost = 0.0

			if Train[i][-1] == 1:
				delX = -1
			else:
				delX = 1

			for j in xrange(feature):
				cost = cost + Weight[j]*Train[i][j]*delX

			if cost <= 0:
				if Train[i][-1] == 1: # W(t+1) = W(t) + p*X
					Weight[0] = Weight[0] + p*Train[i][0]
					Weight[1] = Weight[1] + p*Train[i][1]

			if cost >= 0:
				if Test[i][-1] == 2: # W(t+1) = W(t) - p*X
					Weight[0] = Weight[0] - p*Train[i][0]
					Weight[1] = Weight[1] - p*Train[i][1]

		if count >= iteration:
			break

		#print Weight

	count = testModel(Weight)
	w = copy.deepcopy(Weight)
	Weight = copy.deepcopy(temp)
	return count,w


def pocketAlgo(iteration):
	global Weight,WS,hs,count
	WS,count,hs = [],0,0
	temp = copy.deepcopy(Weight)
	while (True):
		Y = []
		count += 1
		for i in xrange(len(Train)):
			if update(i,Train):
				Y.append(i)

		#print "Y = ",len(Y)

		for i in Y:
			if Train[i][-1] == 1:
				Weight[0] = Weight[0] + p*Train[i][0]
				Weight[1] = Weight[1] + p*Train[i][1]

			else:
				Weight[0] = Weight[0] - p*Train[i][0]
				Weight[1] = Weight[1] - p*Train[i][1]

			h = testModel(Weight)
			if h>hs: # Update hs,WS
				hs = h
				WS = copy.deepcopy(Weight)

		if len(Y) == 0 or count >= iteration:
			break

	count = testModel(WS)
	w = copy.deepcopy(WS)
	Weight = copy.deepcopy(temp)
	return count,w


def makePlot():
	X,Y,X1,Y1,X2,Y2 = [],[],[],[],[],[]
	for i in xrange(sampleSize):
		X.append(Train[i][0])
		Y.append(Train[i][1])

	X1 = X[0:50]
	Y1 = Y[0:50]
	X2 = X[50:100]
	Y2 = Y[50:100]

	plt.figure(0)
	plt.subplot(211)
	class1, = plt.plot(X1,Y1, 'ro')
	class2, = plt.plot(X2,Y2, 'bo')
	plt.axis([0, 12, 0, 15])
	plt.legend([class1, class2], ['Class 1','Class 2'])

	X,Y,X1,Y1,X2,Y2 = [],[],[],[],[],[]
	for i in xrange(sampleSize):
		X.append(Test[i][0])
		Y.append(Test[i][1])

	X1 = X[0:50]
	Y1 = Y[0:50]
	X2 = X[50:100]
	Y2 = Y[50:100]

	plt.subplot(212)
	class1, = plt.plot(X1,Y1, 'ro')
	class2, = plt.plot(X2,Y2, 'bo')
	plt.axis([0, 12, 0, 15])
	plt.legend([class1, class2], ['Class 1','Class 2'])

	plt.suptitle("Train Data (Up) - Test Data (Down)", fontsize=16)
	plt.show()


def linePlot(matrix,w,str1):
	X,Y,X1,Y1,X2,Y2 = [],[],[],[],[],[]
	for i in xrange(sampleSize):
		X.append(matrix[i][0])
		Y.append(matrix[i][1])

	X1 = X[0:50]
	Y1 = Y[0:50]
	X2 = X[50:100]
	Y2 = Y[50:100]

	class1, = plt.plot(X1,Y1, 'ro')
	class2, = plt.plot(X2,Y2, 'bo')

	X,Y = [],[]
	X = np.linspace(0,12,20)
	Y = -(w[0]*X+w[2])/w[1]

	plt.plot(X,Y, "g-")

	plt.axis([0, 12, 0, 15])
	plt.legend([class1, class2], ['Class 1','Class 2'])
	plt.suptitle(str1, fontsize=16)
	plt.show()


def plotting():
	makePlot()
	count1,w = perceptron(100)
	linePlot(Test,w,"Perceptron")
	count2,w = rewardPunishment(100)
	linePlot(Test,w,"Reward- Punishment")
	count3,w = pocketAlgo(100)
	linePlot(Test,w,"Pocket Algorithm")


def run():
	fileInput()
	print "Iteration \t Success - Percept, Reward, Pocket"
	for x in xrange(1,16):
		count1,w = perceptron(x)
		count2,w = rewardPunishment(x)
		count3,w = pocketAlgo(x)

		print ("%5d \t %10d \t %5d \t %5d") % (x,count1,count2,count3)

	for x in xrange(20,101,10):
		count1,w = perceptron(x)
		count2,w = rewardPunishment(x)
		count3,w = pocketAlgo(x)

		print ("%5d \t %10d \t %5d \t %5d") % (x,count1,count2,count3)

	plotting()


if __name__ == '__main__':
	run()