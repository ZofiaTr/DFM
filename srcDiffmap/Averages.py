"""
Averages

import Averages as av
import numpy as np

position=av.Average()
x=np.random.randn(1000)
for i in range(0, len(x)):
    position.addSample(x[i])
    #print x[i]
print position.getAverage()
print np.mean(x)

"""


import numpy as np

class Average():

    def __init__(self, initialEntry, blockSize=10000):

        self.initialEntry=initialEntry
        self.currentBlockIndex=0
        self.blockSize=blockSize
        self.currentIndexInCurrentBlock=0

        self.currentBlockSum=initialEntry
        self.sumOfBlockAverages=initialEntry

    def addSample(self, T):

        self.currentBlockSum=self.currentBlockSum+T
        self.currentIndexInCurrentBlock=self.currentIndexInCurrentBlock+1

        if (self.currentIndexInCurrentBlock==self.blockSize):

            #// add the block average to the sum of block averages
            self.currentBlockAverage= self.currentBlockSum/float(self.blockSize)
            self.sumOfBlockAverages=self.sumOfBlockAverages+self.currentBlockSum/float(self.blockSize)

           #// begin a new block
            self.currentBlockSum=0.0*self.initialEntry
            self.currentIndexInCurrentBlock=0
            self.currentBlockIndex=self.currentBlockIndex+1



    def getAverage(self):

        ans=0

        numberOfSamples=self.blockSize*self.currentBlockIndex+self.currentIndexInCurrentBlock

        if (numberOfSamples>0) :

            currentAverage= self.currentBlockSum/float(numberOfSamples)

            if (self.currentBlockIndex>0):
                currentAverage=currentAverage + self.sumOfBlockAverages/float(self.currentBlockIndex)
                tmp=(self.currentIndexInCurrentBlock*self.sumOfBlockAverages)
                tmp=tmp/float(numberOfSamples)/float(self.currentBlockIndex)
                currentAverage=currentAverage - tmp


            ans= currentAverage

        return ans

    def clear(self):
        self.currentBlockIndex=0
        self.blockSize=10000
        self.currentIndexInCurrentBlock=0
        self.currentBlockSum=self.initialEntry
        self.sumOfBlockAverages=self.initialEntry
