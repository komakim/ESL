# -*- coding: utf-8 -*-

import math
from matplotlib import pyplot as plt
import numpy as np

class Adaboost:
    def __init__(self,varData,objData):
        self.__varData = varData
        self.__objData = objData
        self.__numData = len(varData)
        self.__w = np.array([1/self.__numData]*self.__numData)


    def learn(self,num):
        self.__learnNum = num
        self.__theta = np.array([0.0]*num)
        self.__phi = np.array([0.0]*num)
        self.__phiIndex = np.array([-1]*num)

        for i in range(num):
            tResult = self.__learnPhi()

            self.__phi[i] = tResult[1]
            self.__phiIndex[i] = tResult[0]


            tR = tResult[2]
            self.__theta[i] = 0.5*math.log((1-tR)/tR)

            self.__updateW()

        print(self.__w)
        print("^W")
        print(self.__theta)
        print("^Theta")
        print(self.__phi)
        print("^Phi")
        print(self.__phiIndex)
        print("^phiIndex")


    def __learnPhi(self):
        tBestPhi = [-1,-1,-1]
        for i in range(self.__numData):
            for j in range(len(self.__varData[0])):
                tTemp = self.__errorRate(j,self.__varData[i,j])
                if (tBestPhi[0] < 0)or(tTemp < tBestPhi[2]):
                    tBestPhi = [j,self.__varData[i,j],tTemp]

        return tBestPhi

    def __errorRate(self,index,border):
        error = 0.0
        for i in range(self.__numData):
            error = error + self.__w[i]*(1-(self.__calcPhi(index,border,self.__varData[i,index])*self.__objData[i]))
        return error*0.5


    def __calcPhi(self,index,border,data):
        if ((index == 0)and(data <= border))or((index == 1)and(data >= border)):
            return 1
        else:
            return -1


    def __ef(self,index):
        out = 0.0
        for k in range(self.__learnNum):

            if self.__phiIndex[k] == -1:
                break
            elif ((self.__phiIndex[k] == 0)and(self.__varData[index,0] <= self.__phi[k]))or((self.__phiIndex[k] == 1)and(self.__varData[index,1] >= self.__phi[k])):
                out = out + self.__theta[k]
            else:
                out = out - self.__theta[k]

        return out



    def __updateW(self):
        sum = 0.0
        for i in range(self.__numData):
            sum = sum + math.exp(-1*self.__ef(i)*self.__objData[i])
        for j in range(self.__numData):
            self.__w[j] = math.exp(-1*self.__ef(j)*self.__objData[j])/sum

    def plot(self,fineness):
        plt.figure()
        plt.hold(True)
        plt.title('decision_kabu(AdaBoost)')
        plt.xlabel('x(1)')
        plt.ylabel('x(2)')

        x1Limit = 1.2
        x2Limit = 3.5
        for i in range(fineness):
            for j in range(fineness):
                if self.__isBorder([(j*x1Limit/fineness),(i*x2Limit/fineness)]):
                    plt.scatter((j*x1Limit/fineness),(i*x2Limit/fineness),marker="+", c='y')

        for i in range(self.__numData):
            if self.__objData[i] == -1:
                plt.scatter(self.__varData[i,0],self.__varData[i,1], c='blue')
            else:
                plt.scatter(self.__varData[i,0],self.__varData[i,1], c='r')


    def __isBorder(self,coordination):
        out = 0.0
        for k in range(self.__learnNum):
            if self.__phiIndex[k] == -1:
                break
            elif ((self.__phiIndex[k] == 0)and(coordination[0] <= self.__phi[k]))or((self.__phiIndex[k] == 1)and(coordination[1] >= self.__phi[k])):
                out = out + self.__theta[k]
            else:
                out = out - self.__theta[k]
        if (out >= 0 ):
            return True
        else:
            return False
