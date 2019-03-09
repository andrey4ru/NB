import numpy as np
import pandas as pd
import codecs
import re
import Naive_Bayes as Bayes
import ROC


filename = 'SMSSpamCollection1.txt'
# fileObj = codecs.open(filename, "r", "utf_8_sig")   # read data from file

# dictionaries
train_HAM = []  # 0 - HAM
train_SPAM = []  # 1 - SPAM
control_set = []  # control set
label = []


with codecs.open(filename, "r", "utf_8_sig") as lines:  # Read lines with utf_8 coding
    for text in lines:
        class_SMS = text[-3]  # class of SMS on 3d position from end
        text = re.sub(r'[^\w\s]', '', text)  # Replace all symbol that not character, number or space for ''
        text = text.lower()  # to lower case
        text = text.replace('0\r\n', "")  # replace '\n', '\t' for ''
        text = re.sub(r'\d+', '1', text)  # replace all numbers for '1'
        text = text.split(" ")  # split by ' '

        if class_SMS == '0' and len(train_HAM) < 3000:  # 3000 examples for HAM train set
            train_HAM.extend(text)
        elif class_SMS == '1' and len(train_SPAM) < 464:  # 464 examples for SPAM train set
            train_SPAM.extend(text)
        else:
            control_set.append(text)
            label.append(class_SMS)


model = Bayes.Naive_Bayes()  # create model
model.Train(train_SPAM, train_HAM)  # train model
prob = model.Predict(control_set)  # predict probability

ROC.ROC(prob, label, "1")  # build ROC curve

