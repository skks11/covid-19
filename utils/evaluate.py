import numpy as np
from tqdm import tqdm

class evaluater:
    def __init__(self,emb_file):
        self.l2l = '../data/l2l.txt'
        self.p2l = '../data/p2l.txt'
        self.doublelink = '../data/doublelink_hk.txt'
        self.emb_file = emb_file
        self.emb = {}
        self.positive = []
        self.negtive = []
        

    def cosine_similarity(self,a, b):
        from numpy import dot
        from numpy.linalg import norm
        ''' cosine similarity; can be used as score function; vector by vector; 
            If consider similarity for all pairs,
            pairwise_similarity() implementation may be more efficient
        '''
        a = np.reshape(a,-1)
        b = np.reshape(b,-1)
        if norm(a)*norm(b) == 0:
            return 0.0
        else:
            return dot(a, b)/(norm(a)*norm(b))

    def get_neg(self):
        for i in range(1017):
            for j in range(1017):
                if [i+1,j+1] not in self.positive:
                    if i+1 in self.emb.keys() and j+1 in self.emb.keys():
                        self.negtive.append([i+1,j+1])
        
    def get_pos(self):
        f = open(self.doublelink,'r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            self.positive.append([int(line[0]),int(line[1])])
        f.close()

    def load_embedding(self):
        f = open(self.emb_file,'r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            if len(line) < 5:
                continue
            # print(line)s

            # self.emb[int(line[0][1:])] = list(map(lambda x: float(x),line[1:]))
            if 'node2vec' in self.emb_file:
                self.emb[int(line[0])] = list(map(lambda x: float(x),line[1:]))
            else:
                self.emb[int(line[0][1:])] = list(map(lambda x: float(x),line[1:]))
                
        f.close()

    def compare_similarity(self):
        sim_file = open('../res/sim.txt','a',encoding='utf-8')
        pos = []
        neg = []
        self.load_embedding()
        self.get_pos()
        self.get_neg()
        for pair in self.positive:
            pos1,pos2 = pair[0],pair[1]
            pos.append(self.cosine_similarity(self.emb[pos1],self.emb[pos2]))
        for pair in self.negtive:
            neg1,neg2 = pair[0],pair[1]
            neg.append(self.cosine_similarity(self.emb[neg1],self.emb[neg2]))
        sim_file.write(self.emb_file+'\n')
        print('pos: {}   neg:{}'.format(np.mean(pos),np.mean(neg)))
        sim_file.write(str(np.mean(pos))+' '+str(np.mean(neg))+'\n')
        sim_file.close()

if __name__ == '__main__':
    embs = ['../emb/node2vec.txt','../emb/HIN2vec/node.txt','../emb/Metapath2vec/covid-plp.txt','../emb/HeGAN/covid_dis.emb','../emb/HeGAN/covid_gen.emb']
    for emb in embs:
        print('processing '+emb)
        E = evaluater(emb)
        E.compare_similarity()

        

