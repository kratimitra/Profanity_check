#Installing the required libraries and modules
%pip install better_profanity
from better_profanity import profanity
%pip install profanity_check 
from profanity_check import predict, predict_prob
import numpy as np

#Importing input data file of Twitter Tweets
from google.colab import files
uploaded = files.upload()
file_name = "data.txt"
data = uploaded[file_name].decode("utf-8").split("\r\n")

# Here I am using profanity_check library of Python3.
# It uses a linear SVM model trained on 200k human-labeled samples of clean and profane text strings.

#Taking each sentence of the twitter file
for i in range(len(data)):
  data[i] = data[i].split(".")

  #The function predict_prob() takes an array thus converting the sentence input into numpy array
  my_string = np.array(data[i])
  print(my_string)

  #print(type(data[i]))

  #The function predict_prob() returns the probability how each string is offensive.
  output = predict_prob(my_string)
  #Printing The Output
  print(output)



  
