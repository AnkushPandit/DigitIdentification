
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('C:\opencv\sources\samples\data\digits.png')
#cv2.imshow('img',img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)

# Now we prepare train_data and test_data.
train = x[:,:90].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,90:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,450)[:,np.newaxis]
test_labels = np.repeat(k,50)[:,np.newaxis]

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create() #cv2.KNearest() is old
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=5) #find_nearest

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print (accuracy)

#testing a new image
new_test=cv2.imread("test_example.jpg")

#pre-processing the image
gray2=cv2.cvtColor(new_test,cv2.COLOR_BGR2GRAY)
cv2.imshow('img',gray2)
gray2=cv2.resize(gray2,(20,20))
xy=np.array(gray2)
xy=xy.reshape(-1,400).astype(np.float32)


ret,result,neighbours,dist = knn.findNearest(xy,k=10)
print(format(result))
#print(format(neighbours))
#print(format(dist))

