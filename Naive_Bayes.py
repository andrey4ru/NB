import math


class Naive_Bayes:

    __model = {}
    __prob_of_Spam = 0.16  # prior probability of spam
    __prob_of_Ham = 0.84  # prior probability of ham

    def Train(self, SPAM_set, HAM_set):  # calculate number of words
        SPAM = {}
        for i in range(len(SPAM_set)):
            if SPAM_set[i] in SPAM:  # incrementing counter if word in spam dict
                SPAM[SPAM_set[i]] += 1
            else:
                SPAM[SPAM_set[i]] = 1  # add new word to dictionary

        HAM = {}
        for i in range(len(HAM_set)):
            if HAM_set[i] in HAM:  # incrementing counter if word in ham dict
                HAM[HAM_set[i]] += 1
            else :
                HAM[HAM_set[i]] = 1  # add new word to dictionary

        self.__model["SPAM"] = SPAM
        self.__SPAM_count = len(SPAM_set)  # set spam counter to length of spam set
        self.__model["HAM"] = HAM
        self.__HAM_count = len(HAM_set)  # set ham counter to length of ham set

    def Predict(self, predict_set):  # predict probability
        Prob=[]
        prob_SPAM=[]
        prob_HAM=[]
        for i in range(len(predict_set)):
            prob = math.log(self.__prob_of_Spam)
            for j in range(len(predict_set[i])):
                if predict_set[i][j] in self.__model["SPAM"]:
                    # calculate probability of spam
                    prob += math.log((self.__model["SPAM"][predict_set[i][j]] + 1)/(self.__SPAM_count + 2))
                else:
                    prob += math.log(1/(self.__SPAM_count+2))  # if word not in dictionary
            prob_SPAM.append(prob)

        for i in range(len(predict_set)):
            prob = math.log(self.__prob_of_Ham)
            for j in range(len(predict_set[i])):
                if predict_set[i][j] in self.__model["HAM"]:
                    # calculate probability of ham
                    prob += math.log((self.__model["HAM"][predict_set[i][j]] + 1) / (self.__HAM_count + 2))
                else:
                    prob += math.log(1/(self.__HAM_count+2))  # if word not in dictionary
            prob_HAM.append(prob)

        for i in range(len(prob_SPAM)):
            Prob.append(max(prob_SPAM[i] - prob_HAM[i], 0))  # return probability of spam
        return Prob

