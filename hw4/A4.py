import numpy as np
import pandas as pd
from sklearn import neural_network
from sklearn.metrics import accuracy_score
from random import randrange
import utils
# import matplotlib.pyplot as plt


#Perfect Instances
five =  [0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0]
two = [0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0]
patterns = [five,two]

def loadGeneratedData(path):
  test_data = pd.read_csv(path)
  test_y = list(test_data.iloc[:,-1]) # Labels

  test_X_df = test_data.iloc[:,:-1] # Features
  test_X = []
  # Convert features from a dataframe to a list
  for idx, row in test_X_df.iterrows(): 
    test_X.append(list(row))

  return test_X, test_y

def distort_input(instance, percent_distortion):
    #percent distortion should be a float from 0-1
    distorted = []
    for i in range(len(instance)):
      if np.random.randint(0,100) < percent_distortion * 100:
        if instance[i] == 0:
          distorted.append(1)
        else:
          distorted.append(0)
      else:
        distorted.append(instance[i])
    return distorted


class HopfieldNetwork:
  def __init__(self, size):
    self.h = np.zeros([size,size])

  def addSinglePattern(self, p):
    #Update the hopfield matrix using the passed pattern
    n = len(p)
    for i in range(n):
      for j in range(n):
        if (i != j):
          if (p[i] == p[j]):
            self.h[i][j] += 1
          else:
            self.h[i][j] -= 1

  def fit(self, patterns):
		# for each pattern
		# Use your addSinglePattern function to learn the final h matrix
    for pattern in patterns:
      self.addSinglePattern(pattern)

  def retrieve(self, inputPattern):
		#Use your trained hopfield network to retrieve and return a pattern based on the
		#input pattern.
		#HopfieldNotes.pdf on canvas is a good reference for asynchronous updating which
		#has generally better convergence properties than synchronous updating.

    order = np.arange(len(inputPattern))
    np.random.shuffle(order)
    converged = False
    while (not converged):
      changed = False
      for node_idx in order:
        node = order[node_idx]
        dot_product = np.dot(np.array(self.h[node]), inputPattern)
        if ((dot_product >= 0 and inputPattern[node] == 1) or (dot_product < 0 and inputPattern[node] == 0)):
          continue
        else:
          changed = True
          if (dot_product >=0):
            inputPattern[node] = 1
          else:
            inputPattern[node] = 0
      if (not changed):
        converged = True
    
    return inputPattern

  def classify(self, inputPattern):
		#Classify should consider the input and classify as either, five or two
		#You will call your retrieve function passing the input
		#Compare the returned pattern to the 'perfect' instances
		#return a string classification 'five', 'two' or 'unknown'

    attractor = self.retrieve(inputPattern)

    dist_two = np.linalg.norm(np.array(two)-np.array(attractor))
    dist_five = np.linalg.norm(np.array(five)-np.array(attractor))

    if (dist_two == 0 ):
      return 'two'
    elif (dist_five == 0):
      return 'five'
    else:
      return 'unknown'





if __name__ == "__main__":
  hopfieldNet = HopfieldNetwork(25)

  ##############
  ### PART 2 ###
  ##############

  hopfieldNet.fit(patterns)

  test_X, test_y = loadGeneratedData('mihe1609-TrainingData.csv')
  n = len(test_y)

  # Classify test data, report accuracy
  correct_classifications = 0
  for i in range(n):
    pred = hopfieldNet.classify(test_X[i])
    true = test_y[i]
    if (true == pred):
      correct_classifications += 1
    else :
      print("Incorrectly classified data point", i, " as ", pred) # Usually 0, 1, and 5

  print("Hopfield net accuracy: ", correct_classifications / n)
  
  ##############
  ### PART 3 ###
  ##############

  mlp = neural_network.MLPClassifier()
  mlp.fit(patterns, ['five', 'two'])
  y_pred = mlp.predict(test_X)
  print("MLP accuracy: ", accuracy_score(y_pred, test_y))

  ##############
  ### PART 4 ###
  ##############

  distorted = []
  rates = [k / 100 for k in range(51)]
  hopfield_accuracies = []
  mlp_accuracies = []

  for rate in rates:
    hopfieldNet = HopfieldNetwork(25)
    hopfieldNet.fit(patterns)

    mlp = neural_network.MLPClassifier()
    mlp.fit(patterns, ['five', 'two'])

    distored_X = [distort_input(instance, rate) for instance in test_X]

    # Classify data using hopfield network
    correct_classifications = 0
    for i in range(n):
      pred = hopfieldNet.classify(distored_X[i])
      true = test_y[i]
      if (true == pred):
        correct_classifications += 1
    
    hopfield_accuracies.append(correct_classifications / n)

    # Classify data using mlp
    mlp_pred = mlp.predict(distored_X)
    mlp_accuracies.append(accuracy_score(mlp_pred, test_y))

  # fig, ax = plt.subplots(1,1)
  # ax.plot(rates, hopfield_accuracies, label="Hopfield network accuracy")
  # ax.plot(rates, mlp_accuracies, label="MLP accuracy")
  # ax.set_xlabel("Distortion rate")
  # ax.set_ylabel("Accuracy")
  # ax.set_title("Accuracy vs. Disortion rate")
  # ax.legend()
  # plt.show()

  ##############
  ### PART 5 ###
  ##############

  new_X, new_y = loadGeneratedData('NewInput.csv')
  full_X = new_X + test_X
  full_y = new_y + test_y

  rates = [k / 100 for k in range(51)]
  layers=[1, 2, 3, 4, 5]
  # fig, ax = plt.subplots(1,1)

  for n_layers in layers:
    mlp_accuracies = []
    distorted = []

    tup = tuple([100 for i in range(n_layers)])
    mlp = neural_network.MLPClassifier(hidden_layer_sizes=tup)
    mlp.fit(patterns, ['five', 'two'])

    for rate in rates:
      distored_X = [distort_input(instance, rate) for instance in full_X]

      # Classify data using mlp
      mlp_pred = mlp.predict(distored_X)
      mlp_accuracies.append(accuracy_score(mlp_pred, full_y))

    # ax.plot(rates, mlp_accuracies, label="n_layers =" + str(n_layers))

  # ax.set_xlabel("Distortion rate")
  # ax.set_ylabel("Accuracy")
  # ax.set_title("Accuracy vs. Disortion rate")
  # ax.legend()
  # plt.show()
