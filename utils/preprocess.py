import numpy as np
from tqdm import tqdm
class preprocesser:
    def __init__(self,outfile,emb_file):
        self.mapping = {}
        self.l2l = '../data/l2l.txt'
        self.p2l = '../data/p2l.txt'
        self.doublelink = '../data/doublelink_hk.txt'
        self.emb_file = emb_file
        self.emb = {}
        self.attr = {}
        self.outfile = outfile
        self.positive = []
        self.negtive = []
        self.patient_num = 1017
        self.location_num = 2388
        self.imported = []
    
    def get_import(self):
        f = open('../data/import.txt','r',encoding='utf-8')
        for line in f:
            self.imported.append(int(line.strip()))

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
    
    def load_attr(self):
        f = open('../data/feature_attr_only_hk.txt','r',encoding='utf-8')
        tmp = []
        for line in f:
            line = line.strip().split()
            attrs  = list(map(lambda x: int(x),line))
            tmp.append(attrs)
        f.close()
        tmp = np.asarray(tmp)
        tmp = tmp / tmp.max(axis=0)   #normalize
        # print(tmp[:2])
        for i in range(len(tmp)):
            self.attr[i+1] = tmp[i]

    def load_embedding(self):
        if 'LINE' in self.emb_file:
            import pickle
            f = open(self.emb_file,'rb')
            self.emb = pickle.load(f)
            # print(len(self.emb[1]))
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
        print('negative sampling......')
        self.get_import()
        # 1000 * 50 个负例
        # for i in range(self.patient_num):
        #     if i+1 not in self.emb.keys():
        #         continue
        #     cnt = 0
        #     while cnt < 20:
        #         j = np.random.randint(1,self.patient_num+1)
        #         if i+1 in self.imported and j in self.imported:
        #             continue
        #         if [i+1,j] not in self.positive:
        #             if j in self.emb.keys():
        #                 self.negtive.append([i+1,j])
        #                 cnt += 1

        # 枚举出所有负例
        for i in tqdm(range(1,self.patient_num+1)):
            if i not in self.emb.keys():
                continue
            for j in range(1,self.patient_num+1):
                if j not in self.emb.keys():
                    continue
                if i in self.imported and j in self.imported:
                    continue
                if [i,j] not in self.positive:
                    self.negtive.append([i,j])

        
    def get_pos(self):
        f = open(self.doublelink,'r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            self.positive.append([int(line[0]),int(line[1])])
        f.close()
    
    def get_train_file_attr_only(self):
        print('synthesis training data......')
        self.load_embedding()
        self.get_pos()
        self.get_neg()
        # train = []
        fout = open(self.outfile,'w',encoding='utf-8')
        for pair in self.positive:
            tmp = []
            pos1,pos2 = pair[0],pair[1]
        
            sim2 = self.cosine_similarity(self.attr[pos1],self.attr[pos2])
            tmp.append(sim2)
        
            tmp = tmp + self.attr[pos1].tolist() +self.attr[pos2].tolist() + [1]
                
            
            # train.append(tmp)
            tmp = list(map(lambda x: str(x),tmp))
            fout.write(' '.join(tmp)+'\n')
            
        

        for pair in self.negtive:
            tmp = []
            neg1,neg2 = pair[0],pair[1]
            
            sim2 = self.cosine_similarity(self.attr[pos1],self.attr[pos2])
            tmp.append(sim2)
           
            tmp = tmp  + self.attr[neg1].tolist() +self.attr[neg2].tolist() + [0]
            
            # train.append(tmp)
            tmp = list(map(lambda x: str(x),tmp))
            fout.write(' '.join(tmp)+'\n')
        # print(len(train[0]))   

        fout.close()     
    

    def get_train_file_with_attr(self):
        print('synthesis training data......')
        self.load_embedding()
        self.get_pos()
        self.get_neg()
        # train = []
        fout = open(self.outfile,'w',encoding='utf-8')
        for pair in self.positive:
            tmp = []
            pos1,pos2 = pair[0],pair[1]
            sim = self.cosine_similarity(self.emb[pos1],self.emb[pos2])
            tmp.append(sim)
        
            sim2 = self.cosine_similarity(self.attr[pos1],self.attr[pos2])
            tmp.append(sim2)
            if 'LINE' in self.emb_file:
                tmp = tmp + self.emb[pos1].tolist() + self.emb[pos2].tolist() + self.attr[pos1].tolist() +self.attr[pos2].tolist() + [1]
                
            else:
                tmp = tmp + self.emb[pos1] + self.emb[pos2] + self.attr[pos1].tolist() +self.attr[pos2].tolist() + [1]
            # train.append(tmp)
            tmp = list(map(lambda x: str(x),tmp))
            fout.write(' '.join(tmp)+'\n')
            
        

        for pair in self.negtive:
            tmp = []
            neg1,neg2 = pair[0],pair[1]
            sim = self.cosine_similarity(self.emb[neg1],self.emb[neg2])
            tmp.append(sim)
            sim2 = self.cosine_similarity(self.attr[pos1],self.attr[pos2])
            tmp.append(sim2)
            if 'LINE' in self.emb_file:
                tmp = tmp + self.emb[neg1].tolist() + self.emb[neg2].tolist() + self.attr[neg1].tolist() +self.attr[neg2].tolist() + [0]
            else:
                tmp = tmp + self.emb[neg1] + self.emb[neg2]  + self.attr[neg1].tolist() +self.attr[neg2].tolist() + [0]
            # train.append(tmp)
            tmp = list(map(lambda x: str(x),tmp))
            fout.write(' '.join(tmp)+'\n')
        # print(len(train[0]))   

        fout.close()     
    
    def get_train_file(self,with_attr=False):
        print('synthesis training data......')
        self.load_embedding()
        self.get_pos()
        self.get_neg()
        # train = []
        fout = open(self.outfile,'w',encoding='utf-8')
        for pair in self.positive:
            tmp = []
            pos1,pos2 = pair[0],pair[1]
            sim = self.cosine_similarity(self.emb[pos1],self.emb[pos2])
            tmp.append(sim)
            if with_attr:
                sim2 = self.cosine_similarity(self.attr[pos1],self.attr[pos2])
                tmp.append(sim2)
            if 'LINE' in self.emb_file:
                # if  with_attr:
                #     tmp = tmp + self.emb[pos1].tolist() + self.emb[pos2].tolist() + self.attr. [1]
                # else:
                tmp = tmp + self.emb[pos1].tolist() + self.emb[pos2].tolist() + [1]
            else:
                tmp = tmp + self.emb[pos1] + self.emb[pos2] + [1]
            # train.append(tmp)
            tmp = list(map(lambda x: str(x),tmp))
            fout.write(' '.join(tmp)+'\n')
            
        

        for pair in self.negtive:
            tmp = []
            neg1,neg2 = pair[0],pair[1]
            sim = self.cosine_similarity(self.emb[neg1],self.emb[neg2])
            tmp.append(sim)
            if 'LINE' in self.emb_file:
                tmp = tmp + self.emb[neg1].tolist() + self.emb[neg2].tolist() + [0]
            else:
                tmp = tmp + self.emb[neg1] + self.emb[neg2] + [0]
            # train.append(tmp)
            tmp = list(map(lambda x: str(x),tmp))
            fout.write(' '.join(tmp)+'\n')
        # print(len(train[0]))   

        fout.close()     
        # np.save(self.outfile,train)
        

        




if __name__ == '__main__':
    # P = preprocesser('../data/l2l.txt','../data/p2l.txt','../data/edge.txt')
    # P.get_edge()

    # P = preprocesser('../data/l2l.txt','../data/p2l.txt','../data/edge_nonHINE.txt')
    # P.get_non_HIN()


    # datasets = [['../data/train/LINE.txt','../emb/LINE.pkl'],
    # ['../data/train/node2vec.txt','../emb/node2vec.txt'],
    # ['../data/train/HIN2vec.txt','../emb/HIN2vec/node.txt'],
    # ['../data/train/metapath2vec.txt','../emb/Metapath2vec/covid-plp.txt'],
    # ['../data/train/HeGANdis.txt','../emb/HeGAN/covid_dis.emb'],
    # ['../data/train/HeGANgen.txt','../emb/HeGAN/covid_gen.emb'],
    # ['../data/train/HeGANmean_with_attr.txt','../emb/HeGAN/covid_mean.emb']]
    
   
    # for dataset in datasets:
    #     print('processing '+dataset[1])
    #     P = preprocesser(dataset[0],dataset[1])
    #     P.get_train_file()
   

    datasets = [['../data/train/LINE_with_attr.txt','../emb/LINE.pkl'],
    ['../data/train/node2vec_with_attr.txt','../emb/node2vec.txt'],
    ['../data/train/HIN2vec_with_attr.txt','../emb/HIN2vec/node.txt'],
    ['../data/train/metapath2vec_with_attr.txt','../emb/Metapath2vec/covid-plp.txt'],
    ['../data/train/HeGANdis_with_attr.txt','../emb/HeGAN/covid_dis.emb'],
    ['../data/train/HeGANgen_with_attr.txt','../emb/HeGAN/covid_gen.emb'],
    ['../data/train/HeGANmean_with_attr.txt','../emb/HeGAN/covid_mean.emb']]
    
    # P = preprocesser('../data/train/node2vec.txt','../data/node2vec.txt')
    for dataset in [datasets[3]]:
    # for dataset in datasets:
        print('processing '+dataset[1])
        P = preprocesser(dataset[0],dataset[1])
        P.load_attr()
        P.get_train_file_with_attr()

    # datasets = [['../data/train/LINE_attr_only.txt','../emb/LINE.pkl'],
    # ['../data/train/node2vec_attr_only.txt','../emb/node2vec.txt'],
    # ['../data/train/HIN2vec_attr_only.txt','../emb/HIN2vec/node.txt'],
    # ['../data/train/metapath2vec_attr_only.txt','../emb/Metapath2vec/covid-plp.txt'],
    # ['../data/train/HeGANdis_attr_only.txt','../emb/HeGAN/covid_dis.emb'],
    # ['../data/train/HeGANgen_attr_only.txt','../emb/HeGAN/covid_gen.emb']]
    
    # # P = preprocesser('../data/train/node2vec.txt','../data/node2vec.txt')
    # for dataset in [datasets[3]]:
    # # for dataset in datasets:
    #     print('processing '+dataset[1])
    #     P = preprocesser(dataset[0],dataset[1])
    #     P.load_attr()
    #     P.get_train_file_attr_only()

