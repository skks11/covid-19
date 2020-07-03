import numpy as np
class preprocesser:
    def __init__(self,outfile,emb_file):
        self.mapping = {}
        self.l2l = '../data/l2l.txt'
        self.p2l = '../data/p2l.txt'
        self.doublelink = '../data/doublelink_hk.txt'
        self.emb_file = emb_file
        self.emb = {}
        self.outfile = outfile
        self.positive = []
        self.negtive = []
        self.patient_num = 1017
        self.location_num = 2388

    def get_mappings(self):
        f = open('../data/node_mappings.txt','w',encoding = 'utf-8')
        for i in range(self.patient_num):
            f.write(str(i+1)+' patient\n')
        for j in range(self.location_num):
            self.mapping[str(j+1)] = str(j+self.patient_num+1)
            f.write(str(j+self.patient_num+1)+' lcoation\n')
        f.close()

    # 注意是否需要双向
    def get_non_HIN(self):
        self.get_mappings()
        fout = open(self.outfile,'w',encoding='utf-8')
        
        f = open(self.l2l,'r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            fout.write(self.mapping[line[0]]+' '+self.mapping[line[1]]+' '+line[2]+'\n')
        f.close()

        f = open(self.p2l,'r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            fout.write(line[0]+' '+self.mapping[line[1]]+' 1'+'\n')
            fout.write(self.mapping[line[1]]+' '+line[0]+' 1'+'\n')
        f.close()
        fout.close()

    ## for OpenHINE
    def get_edge(self):
        self.get_mappings()
        fout = open(self.outfile,'w',encoding='utf-8')

        f = open(self.l2l,'r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            fout.write(self.mapping[line[0]]+' '+self.mapping[line[1]]+' l-l '+line[2]+'\n')
        f.close()

        f = open(self.p2l,'r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            fout.write(line[0]+' '+self.mapping[line[1]]+' p-l '+'1'+'\n')
            fout.write(self.mapping[line[1]]+' '+line[0]+' l-p '+'1'+'\n')
        f.close()

        fout.close()
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
    def load_embedding(self):
        if 'LINE' in self.emb_file:
            import pickle
            f = open(self.emb_file,'rb')
            self.emb = pickle.load(f)
            f.close()
            return

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

    def get_neg(self):
        for i in range(self.patient_num):
            if i+1 not in self.emb.keys():
                continue
            flag = True
            while flag:
                j = np.random.randint(1,self.patient_num+1)
                if [i+1,j] not in self.positive:
                    if j in self.emb.keys():
                        self.negtive.append([i+1,j])
                        flag = False

        
    def get_pos(self):
        f = open(self.doublelink,'r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            self.positive.append([int(line[0]),int(line[1])])
        f.close()
    
    def get_train_file(self):
        self.load_embedding()
        self.get_pos()
        self.get_neg()
        train = []
        for pair in self.positive:
            tmp = []
            pos1,pos2 = pair[0],pair[1]
            sim = self.cosine_similarity(self.emb[pos1],self.emb[pos2])
            tmp.append(sim)
            tmp = tmp + self.emb[pos1] + self.emb[pos2] + [1]
            train.append(tmp)
            # tmp = list(map(lambda x: str(x),tmp))
        

        for pair in self.negtive:
            tmp = []
            neg1,neg2 = pair[0],pair[1]
            sim = self.cosine_similarity(self.emb[neg1],self.emb[neg2])
            tmp.append(sim)
            tmp = tmp + self.emb[neg1] + self.emb[neg2] + [0]
            train.append(tmp)
        # print(train[0])
        # print(train[-1])
        np.save(self.outfile.split('.')[0],train)
        

        




if __name__ == '__main__':
    # P = preprocesser('../data/l2l.txt','../data/p2l.txt','../data/edge.txt')
    # P.get_edge()

    # P = preprocesser('../data/l2l.txt','../data/p2l.txt','../data/edge_nonHINE.txt')
    # P.get_non_HIN()
    datasets = [['../data/train/LINE.txt','../emb/LINE.pkl'],
    ['../data/train/node2vec.txt','../emb/node2vec.txt'],
    ['../data/train/HIN2vec.txt','../emb/HIN2vec/node.txt'],
    ['../data/train/metapath2vec.txt','../emb/Metapath2vec/covid-plp.txt'],
    ['../data/train/HeGANdis.txt','../emb/HeGAN/covid_dis.emb'],
    ['../data/train/HeGANgen.txt','../emb/HeGAN/covid_gen.emb']]
    
    # P = preprocesser('../data/train/node2vec.txt','../data/node2vec.txt')
    for dataset in datasets:
        print('processing '+dataset[1])
        P = preprocesser(dataset[0],dataset[1])
        P.get_train_file()
         


