import random
import numpy as np


# This Env is used for offline training.
class Env:
    def __init__(self, stoptime, filename, epoch, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.__stoptime = stoptime
        self.__filename = filename
        self.__filepath = "/home/netlab/gengnan/ns-allinone-3.26/ns-3.26/scratch/DRLTE/inputs/" + filename + ".txt"
        self.__epoch = epoch
        self.__nodenum = 0
        self.__capacity = 1000.0
        self.__sessnum = 0
        self.__sesspaths = []
        self.__pathnum = []
        self.__sessrate = []
        self.__minrate = 0
        self.__maxrate = self.__capacity
        self.__flowmap = []
        self.__updatenum = 0
        self.readFile()
        self.initFlowMap()

    def readFile(self):
        file = open(self.__filepath, 'r')
        lines = file.readlines()
        file.close()
        sesspath = []
        for i in range(1, len(lines)-2):
            lineList = lines[i].strip().split(',')
            if len(sesspath) != 0 and (int(lineList[1]) != sesspath[0][0] or int(lineList[-2]) != sesspath[0][-1]):
                self.__sesspaths.append(sesspath)
                sesspath = []
            sesspath.append(list(map(int, lineList[1:-1])))
        self.__sesspaths.append(sesspath)
        self.__sessnum = len(self.__sesspaths)
        self.__pathnum = [len(item) for item in self.__sesspaths]
        for item in self.__sesspaths:
            self.__nodenum = max([self.__nodenum, max([max(i) for i in item])])
        self.__nodenum += 1

    def initFlowMap(self):
        for i in range(self.__nodenum):
            self.__flowmap.append([])
            for j in range(self.__nodenum):
                self.__flowmap[i].append([0])

    def getFlowMap(self, action):
        if action == []:
            for item in self.__pathnum:
                action += [round(1.0/item, 3) for j in range(item)]
        #print("action:")
        #print(action)
        subRates = []
        count = 0
        for i in range(self.__sessnum):
            subRates.append([])
            for j in range(self.__pathnum[i]):
                tmp = 0
                if j == self.__pathnum[i] - 1:
                    tmp = self.__sessrate[i] - sum(subRates[i])
                else:
                    tmp = self.__sessrate[i]*action[count]
                count += 1
                subRates[i].append(tmp)

        for i in range(self.__nodenum):
            for j in range(self.__nodenum):
                self.__flowmap[i][j] = 0
        for i in range(self.__sessnum): 
            for j in range(self.__pathnum[i]):
                for k in range(len(self.__sesspaths[i][j])-1):
                    enode1 = self.__sesspaths[i][j][k]
                    enode2 = self.__sesspaths[i][j][k+1]
                    self.__flowmap[enode1][enode2] += subRates[i][j]
    
    def getUtil(self):
        maxutil = max([max(item) for item in self.__flowmap])/self.__capacity
        print("maxutil: %f" % maxutil)

    def update(self, action):
        if self.__updatenum%self.__epoch == 0:
            self.getRates()
            print("sessrates:")
            print(self.__sessrate)
        self.getFlowMap(action)
        self.getUtil()
        self.__updatenum += 1

    def getRates(self):
        self.__sessrate = []
        for i in range(self.__sessnum):
            self.__sessrate.append(random.randint(self.__minrate, self.__maxrate))

    def setRateBound(self, minrate, maxrate):
        self.__minrate = minrate
        self.__maxrate = maxrate
    def showInfo(self):
        print("--------------------------")
        print("----detail information----")
        print("stoptime:%d" % self.__stoptime)
        print("filepath:%s" % self.__filepath)
        print("nodenum:%d" % self.__nodenum)
        print("sessnum:%d" % self.__sessnum)
        print("capacity:%f" % self.__capacity)
        print("pathnum:")
        print(self.__pathnum)
        #print("sessrate:")
        #print(self.__sessrate)
        #print("sesspaths:")
        #print(self.__sesspaths)
        #print("flowmap:")
        #print(self.__flowmap)
        print("--------------------------")
        
if __name__ == "__main__":
    env = Env(1000, "NSF_30_OBL_3_50", 10, 66)
    env.showInfo() #get initial info
    for i in range(100): #update
        env.update([])
    