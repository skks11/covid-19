#!/usr/bin/python
import sys
import numpy as np
import random

class metapathGeneration:
    def __init__(self, pnum, lnum):
        self.pnum = pnum 
        self.lnum = lnum 
        self.p2l = ''
        self.l2l = ''        


        ub = self.load_ub('../data/ub_0.8.train')
        self.get_PP(ub, '../data/metapath/ubu_0.8.txt')
        self.get_PLP(ub, '../data/bca.txt', '../data/metapath/ubcabu_0.8.txt')
        self.get_PLLP()
        
    def get_PLLP(self):
        pl = np.zeros(self.pnum,self.lnum)
        ll = np.zeros(self.lnum,self.lnum)
        p2l = open(self.p2l,'r',encoding='utf-8')
        for line in p2l:
            pid,lid,weight = line.strip().split()
            pl[int(pid)][int(lid)] = float(weight)
        p2l.close()

        l2l = open(self.l2l,'r',encoding='utf-8')
        for line in l2l:
            lid1,lid2,weight = lien.strip().split()
            ll[int(lid1)][int(lid2)] = float(weight)
        l2l.close()
        pllp=(pl.dot(ll.dot(ll.T))).dot(pl.T)

        outfile = open('pllp.txt','w',encoding='utf-8')
        for i in range(pllp.shape[0]):
            for j in range(pllp.shape[1]):
                if i!=j and pllp[i][j] != 0:
                    outfile
            
    def load_ub(self, ubfile):
        ub = np.zeros((self.unum, self.bnum))
        with open(ubfile, 'r') as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                ub[int(user)][int(item)] = 1 
        return ub
    
    def get_UCoU(self, ucofile, targetfile):
        print 'UCoU...'
        uco = np.zeros((self.unum, self.conum))
        with open(ucofile, 'r') as infile:
            for line in infile.readlines():
                u, co, _ = line.strip().split('\t')
                uco[int(u)][int(co)] = 1

        uu = uco.dot(uco.T)
        print uu.shape
        print 'writing to file...'
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(uu[i][j])) + '\n')
                        total += 1
        print 'total = ', total
    
    def get_UU(self, uufile, targetfile):
        print 'UU...'
        uu = np.zeros((self.unum, self.unum))
        with open(uufile, 'r') as infile:
            for line in infile.readlines():
                u1, u2, _ = line.strip().split('\t')
                uu[int(u1)][int(u2)] = 1
        r_uu = uu.dot(uu.T)

        print r_uu.shape
        print 'writing to file...'
        total = 0 
        with open(targetfile, 'w') as outfile:
            for i in range(r_uu.shape[0]):
                for j in range(r_uu.shape[1]):
                    if r_uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(r_uu[i][j])) + '\n')
                        total += 1
        print 'total = ', total
                                                                                                                                     

    def get_UBU(self, ub, targetfile):
        print 'UMU...'

        uu = ub.dot(ub.T)
        print uu.shape
        print 'writing to file...'
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(uu[i][j])) + '\n')
                        total += 1
        print 'total = ', total
    
    def get_BUB(self, ub, targetfile):
        print 'MUM...'
        mm = ub.T.dot(ub)
        print mm.shape
        print 'writing to file...'
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0]):
                for j in range(mm.shape[1]):
                    if mm[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(mm[i][j])) + '\n')
                        total += 1
        print 'total = ', total
    
    

    
    
    
    
    
    

if __name__ == '__main__':
    #see __init__() 
    metapathGeneration(unum=16239, bnum=14284, conum=11, canum=511, cinum=47)
