# USAGE
# python recognize.py --training images/training --testing images/testing

# import the necessary packages
import os
import RV_uniform
# from sklearn.svm import LinearSVC
from sklearn import svm
from imutils import paths
import argparse
import cv2
# To read class from file
import csv
# For plotting
import matplotlib.pyplot as plt
# For array manipulations
import numpy as np
# For saving histogram values
from sklearn.externals import joblib
# Utility Package
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import itertools
from itertools import izip_longest
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('frontalEyes35x16.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
                help="path to the training images")
ap.add_argument("-e", "--testing", required=True,
                help="path to the testing images")
#ap.add_argument("-l", "--imageLabels", help="Path to Image Label Files", required="True")
args = vars(ap.parse_args())

#with open(args['imageLabels'], 'rb') as csvfile:
 #   reader = csv.reader(csvfile, delimiter=' ')
  #  for row in reader:
   #     test_labels = row

#label1 = ['neutral','sad']
# label1 = ['anger','smile']
label1=['AD','BH','BIBH','NE','KR','HS','RD']
#label1=['Adbhut','Bhayanak','Bibhatsa','Hasya','Karun','neutral','Raudra','Shant','Shringar','Veer']
# initialize the local binary patterns descriptor along with
# the data and label lists
data = []
labels = []

def resize1 (image, height, width, max_height, max_width):
    # only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor1 = max_height / float(height)
        # if max_width/float(width) < scaling_factor:
        scaling_factor2 = max_width / float(width)
        # resize image
        image = cv2.resize(image, None, fx=scaling_factor2, fy=scaling_factor1, interpolation=cv2.INTER_AREA)
    return image

def accuracy (test_label, predictions):
    correct = 0
    for x in range(0, len(test_label)):
        if predictions[x] == test_label[x]:
            correct += 1
    return (correct / float(len(test_label))) * 100.0


def plot_confusion_matrix (cm, classes,
                           normalize=False,
                           title='Confusion matrix',
                           cmap=plt.cm.Blues):
    """
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')

def average(X_t):
    temp = []
    izip_longest(*X_t, fillvalue=0)
    temp = [sum(i)/len(i) for i in izip_longest(*X_t, fillvalue=0)]
    return temp


# loop over the training images
for imagePath in paths.list_images(args["training"]):
    # load the image, convert it to grayscale, and describe it
    #print ("imagepath '{}'".format(imagePath))
    im_gray = cv2.imread(imagePath)
    # Convert to grayscale as LBP works on grayscale image
    # im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    height1 = len(im_gray)
    width1 = len(im_gray[0])
    #im_gray = resize1(im_gray,height1,width1,700,600)
    faces = face_cascade.detectMultiScale(im_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = im_gray[y:y + h, x:x + w]
    filename = "cropface.jpg"
    cv2.imwrite(filename, roi_gray)
    im1 = RV_uniform.LBP('cropface.jpg')
    hist = im1.execute()
    Hist = np.concatenate([np.array(i) for i in hist])
    #print "hist:'{}' len(hist):'{}'".format(Hist, len(Hist))
    # extract the label from the image path, then update the
    # label and data lists
    labels.append(os.path.split(os.path.dirname(imagePath))[-1])
    data.append(Hist)

# train a Linear SVM on the data
#model = LinearSVC(C=100.0,random_state=42,multi_class='ovr')
model=svm.SVC(C=2**20,gamma=2**(-8))
model.fit(data, labels)

print "1"
results = []
labels1=[]
# loop over the testing images
for imagePath in paths.list_images(args["testing"]):
    # load the image, convert it to grayscale, describe it,
    # and classify it
    #print ("imagepath '{}'".format(imagePath))
    im_gray = cv2.imread(imagePath)
    # Convert to grayscale as LBP works on grayscale image
    # im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    height2 = len(im_gray)
    width2 = len(im_gray[0])
    #im_gray = resize1(im_gray,height1,width1,700,600)
    faces = face_cascade.detectMultiScale(im_gray, 1.3, 2)
    for (x, y, w, h) in faces:
        roi_gray = im_gray[y:y + h, x:x + w]
    filename = "cropface.jpg"
    cv2.imwrite(filename, roi_gray)
    im1 = RV_uniform.LBP('cropface.jpg')
    hist1 = im1.execute()
    Hist1 = np.concatenate([np.array(i) for i in hist1])
    prediction = model.predict(Hist1.reshape(1, -1))[0]
    labels1.append(os.path.split(os.path.dirname(imagePath))[-1])
    results.append(prediction)
    # display the image and the prediction
    #im = cv2.resize(im_gray, (600, 600))
    cv2.putText(im_gray, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3)
    cv2.imshow("Image", im_gray)
    cv2.waitKey(100)

print len(results), len(labels1)
print "Displaying predictions for {} ** \n".format(results)
acc = accuracy(labels1, results)
print "Displaying accuracy for {} ** \n".format(acc)
conf_metric = confusion_matrix(results, labels1)
plt.figure()
plot_confusion_matrix(conf_metric, classes=label1,
                      title='Confusion matrix, without normalization')
print 'Accuracy Score :', accuracy_score(labels1, results)
print 'Report : '
print classification_report(labels1, results)
plt.show()
