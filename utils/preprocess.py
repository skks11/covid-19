import numpy as np
from tqdm import tqdm

class preprocesser:
    def __init__(self,outfile,emb_file):
        self.mapping = {}
        self.l2l = '../data/l2l.txt'
        self.p2l = '../data/p2l.txt'
        self.ptol = {}
        self.doublelink = '../data/doublelink_hk.txt'
        self.emb_file = emb_file
        self.emb = {}
        self.attr = {}
        self.outfile = outfile
        self.positive = []
        self.negtive = []
        self.patient_num = 2000
        self.location_num = 8233
        self.imported = []   #输入案例
        self.local = []
        self.local2 = []    #没有传染关系的local病例
        self.date = {}          #确诊日期
        self.num_neg = 5

    # def get_local(self):
    def load_p2l(self):
        print('loading p2l......')
        f = open(self.p2l,'r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            if int(line[0]) in self.ptol:
                self.ptol[int(line[0])].append(int(line[1]))
            else:
                self.ptol[int(line[0])] = [int(line[1])]
        for i in range(1,self.patient_num+1):
            if i not in self.ptol:
                self.ptol[i] = []
        f.close()

    def get_type(self):
        f = open('../data/preprocess/node_type.txt','r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            if line[-1] == 'classification_imported':
                self.imported.append(int(line[0]))
            if line[-1] == 'classification_local':
                self.local.append(int(line[0]))
            if line[-1] == 'classification_local2':
                self.local2.append(int(line[0]))

            
        f.close()
    
    def get_date(self):
        f = open('../data/train/feature_attr_only_hk.txt','r',encoding='utf-8')
        for i,line in enumerate(f):
            line = line.strip().split()
            self.date[i+1] = int(line[2])            

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
            fout.write(self.mapping[line[0]]+'\t'+self.mapping[line[1]]+'\tl-l\t'+line[2]+'\t\n')
        f.close()

        f = open(self.p2l,'r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            fout.write(line[0]+'\t'+self.mapping[line[1]]+'\tp-l\t'+'1'+'\t\n')
            fout.write(self.mapping[line[1]]+'\t'+line[0]+'\tl-p\t'+'1'+'\t\n')
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
        f = open('../data/train/feature_attr_only_hk.txt','r',encoding='utf-8')
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

        for i in range(1,self.patient_num+1):               # 无emb节点全0
            if i not in self.emb.keys():
                self.emb[i] = [0 for _ in range(64)]

    def get_neg(self,full=False):
        print('negative sampling......')
        self.get_type()
        self.get_date()
        if not full:
            
            # 1000 * 50 个负例
            for i in tqdm(range(1,self.patient_num+1)):
                if i not in self.emb.keys():
                    continue
                cnt = 0
                while cnt < self.num_neg:
                    j = np.random.randint(1,self.patient_num+1)
                    if i == j:
                        continue
                    if i in self.imported and j in self.imported:
                        continue
                    if abs(self.date[i]-self.date[j]) >= 14:         #确诊日期相差大于两周
                        continue
                    if [i,j] not in self.positive:
                        if j in self.emb.keys():
                            self.negtive.append([i,j])
                            cnt += 1
        else:
        # 枚举出所有负例
            for i in tqdm(range(1,self.patient_num+1)):
                if i not in self.emb.keys():
                    continue
                for j in range(1,self.patient_num+1):
                    if i == j:
                        continue
                    if j not in self.emb.keys():
                        continue
                    if i in self.imported and j in self.imported:    #均为输入案例
                        continue
                    if abs(self.date[i]-self.date[j]) >= 14:         #确诊日期相差大于两周
                        continue
                    if [i,j] not in self.positive:
                        self.negtive.append([i,j])

        
    def get_pos(self):
        f = open(self.doublelink,'r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            if int(line[0]) > self.patient_num or int(line[1]) > self.patient_num:     #存在超过#2000的关系
                continue
            self.positive.append([int(line[0]),int(line[1])])
        f.close()
    

    def get_train_file_attr_only(self):
        self.load_embedding()
        self.get_pos()
        self.get_neg()
        print('synthesis training data......')
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
    
    def load_valid(self):
        
        f = open('../data/preprocess/validset.txt','r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            if int(line[0]) > self.patient_num or int(line[0]) > self.patient_num:
                continue
            self.positive.append([int(line[0]),int(line[1])])
        f.close()
       

    def get_valid_file_with_attr(self):
        self.load_embedding()
        self.load_valid()
        # self.get_neg()
        print('synthesis validation data......')
        # train = []
        fout = open('../data/train/valid.txt','w',encoding='utf-8')
        # fout2 = open('../data/train/mapping.txt','w',encoding='utf-8')

        for pair in self.positive:
            tmp = []
            pos1,pos2 = pair[0],pair[1]
            sim = self.cosine_similarity(self.emb[pos1],self.emb[pos2])
            tmp.append(sim)
        
            sim2 = self.cosine_similarity(self.attr[pos1],self.attr[pos2])
            tmp.append(sim2)
            if 'LINE' in self.emb_file:
                tmp = tmp + self.emb[pos1].tolist() + self.emb[pos2].tolist() + self.attr[pos1].tolist() +self.attr[pos2].tolist() 
                
            else:
                tmp = tmp + self.emb[pos1] + self.emb[pos2] + self.attr[pos1].tolist() +self.attr[pos2].tolist() 
            # train.append(tmp)
            tmp = list(map(lambda x: str(x),tmp))
            fout.write(' '.join(tmp)+'\n')
            # fout2.write(str(pos1)+' '+str(pos2)+'\n')
        fout.close()

    def get_train_file_with_attr(self,full=False):
        self.load_embedding()
        self.get_pos()
        self.get_neg(full)
        print('synthesis training data......')
        # train = []
        fout = open(self.outfile,'w',encoding='utf-8')
        fout2 = open('../data/train/mapping.txt','w',encoding='utf-8')

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
            fout2.write(str(pos1)+' '+str(pos2)+'\n')
        

        for pair in self.negtive:
            tmp = []
            neg1,neg2 = pair[0],pair[1]
            sim = self.cosine_similarity(self.emb[neg1],self.emb[neg2])
            tmp.append(sim)
            sim2 = self.cosine_similarity(self.attr[neg1],self.attr[neg2])
            tmp.append(sim2)
            if 'LINE' in self.emb_file:
                tmp = tmp + self.emb[neg1].tolist() + self.emb[neg2].tolist() + self.attr[neg1].tolist() +self.attr[neg2].tolist() + [0]
            else:
                tmp = tmp + self.emb[neg1] + self.emb[neg2]  + self.attr[neg1].tolist() +self.attr[neg2].tolist() + [0]
            # train.append(tmp)
            tmp = list(map(lambda x: str(x),tmp))
            fout.write(' '.join(tmp)+'\n')
            fout2.write(str(neg1)+' '+str(neg2)+'\n')
        # print(len(train[0]))   
        fout2.close()
        fout.close()     

    def get_predict_file_with_attr(self,full=False):
        self.load_embedding()
        self.get_type()
        print('synthesis training data......')

        # train = []
        fout = open(self.outfile,'w',encoding='utf-8')
        
        

        for pos1 in self.local2[:-160]:
            print(pos1)
            for pos2 in range(1,self.patient_num+1):
                if pos1 == pos2:
                    continue
                
                tmp = []
                sim = self.cosine_similarity(self.emb[pos1],self.emb[pos2])
                tmp.append(sim)
            
                sim2 = self.cosine_similarity(self.attr[pos1],self.attr[pos2])
                tmp.append(sim2)
                if 'LINE' in self.emb_file:
                    tmp = tmp + self.emb[pos1].tolist() + self.emb[pos2].tolist() + self.attr[pos1].tolist() +self.attr[pos2].tolist() + [pos1,pos2]
                    
                else:
                    tmp = tmp + self.emb[pos1] + self.emb[pos2] + self.attr[pos1].tolist() +self.attr[pos2].tolist() + [pos1,pos2]
                # train.append(tmp)
                tmp = list(map(lambda x: str(x),tmp))
                fout.write(' '.join(tmp)+'\n')
                
        fout.close()     
    
    def check_sim(self):
        self.load_attr()
        self.load_embedding()
        self.get_pos()
        self.get_neg()
        pos_sim = []
        neg_sim = []

        for pair in self.positive:
            pos1,pos2 = pair[0],pair[1]
            sim = self.cosine_similarity(self.emb[pos1],self.emb[pos2])
            pos_sim.append(sim)

        for pair in self.negtive:
            neg1,neg2 = pair[0],pair[1]
            sim = self.cosine_similarity(self.emb[neg1],self.emb[neg2])
            neg_sim.append(sim)
        
        pos_avg = np.mean(pos_sim)
        neg_avg = np.mean(neg_sim)
        print('embedding pos similarity avg: {}'.format(pos_avg))
        print('embedding neg similarity avg: {}'.format(neg_avg))

        attr_pos_sim = []
        attr_neg_sim = []

        for pair in self.positive:
            pos1,pos2 = pair[0],pair[1]
            sim = self.cosine_similarity(self.attr[pos1],self.attr[pos2])
            attr_pos_sim.append(sim)

        for pair in self.negtive:
            neg1,neg2 = pair[0],pair[1]
            sim = self.cosine_similarity(self.attr[neg1],self.attr[neg2])
            attr_neg_sim.append(sim)
        
        attr_pos_avg = np.mean(attr_pos_sim)
        attr_neg_avg = np.mean(attr_neg_sim)
        print('attributes pos similarity avg: {}'.format(attr_pos_avg))
        print('attributes neg similarity avg: {}'.format(attr_neg_avg))


        return pos_avg,neg_avg,attr_pos_avg,attr_neg_avg        


    def get_baseline_train_file(self):
        pos_avg,neg_avg,attr_pos_avg,attr_neg_avg = self.check_sim()
        print((pos_avg + neg_avg)/2)
        print((attr_pos_avg + attr_neg_avg)/2)
        self.get_date()
        self.load_p2l()
        # train = []
        print('synthesis training data......')
        fout = open(self.outfile,'w',encoding='utf-8')

        for pair in self.positive:
            tmp = []
            pos1,pos2 = pair[0],pair[1]
            sim = self.cosine_similarity(self.emb[pos1],self.emb[pos2])
            tmp.append(sim)
        
            sim2 = self.cosine_similarity(self.attr[pos1],self.attr[pos2])
            tmp.append(sim2)
            
            common_location = meets = len(set(self.ptol[pos1]) & set(self.ptol[pos2]))
            tmp.append(common_location)
            tmp.append(abs(self.date[pos1]-self.date[pos2]))   # 住院时间差
            tmp.append(1)
            tmp = list(map(lambda x: str(x),tmp))
            fout.write(' '.join(tmp)+'\n')
        
        for pair in self.negtive:
            tmp = []
            neg1,neg2 = pair[0],pair[1]
            sim = self.cosine_similarity(self.emb[neg1],self.emb[neg2])
            tmp.append(sim)
            sim2 = self.cosine_similarity(self.attr[neg1],self.attr[neg2])
            tmp.append(sim2)
            common_location = len(set(self.ptol[neg1]) & set(self.ptol[neg2]))
            tmp.append(common_location)
            tmp.append(abs(self.date[neg1]-self.date[neg2]))   # 住院时间差
            tmp.append(0)
            tmp = list(map(lambda x: str(x),tmp))
            fout.write(' '.join(tmp)+'\n')

        fout.close()

   

    def check_date(self):
        dates = []
        self.get_pos()
        self.get_date()
        cnt = 0
        print(len(self.positive))
        for pair in self.positive:
            tmp = abs(self.date[pair[0]]-self.date[pair[1]])
            if tmp > 14 :
                print(pair)
                cnt += 1
            dates.append(tmp)
            
        print('# of > 14 : {}'.format(cnt))
        print('avg: {}'.format(np.mean(dates)))
        print('max: {}'.format(max(dates)))


    def get_train_file(self,with_attr=False):
        self.load_embedding()
        self.get_pos()
        self.get_neg()
        # train = []
        print('synthesis training data......')
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
    # P = preprocesser('../data/edge.txt','../data/p2l.txt')
    
    # P.get_edge()

    # P = preprocesser('../data/l2l.txt','../data/p2l.txt','../data/edge_nonHINE.txt')
    # P.get_non_HIN()


    # datasets = [['../data/train/LINE.txt','../emb/LINE.pkl'],
    # ['../data/train/node2vec.txt','../emb/node2vec.txt'],
    # ['../data/train/HIN2vec.txt','../emb/HIN2vec/node.txt'],
    # ['../data/train/metapath2vec.txt','../emb/Metapath2vec/covid-plp.txt'],
    # ['../data/train/HeGANdis.txt','../emb/HeGAN/covid_dis.emb'],
    # ['../data/train/HeGANgen.txt','../emb/HeGAN/covid_gen.emb'],
    # ['../data/train/HeGANmean.txt','../emb/HeGAN/covid_mean.emb']]
    
    # for dataset in [datasets[-1],datasets[3]]:
    # # for dataset in datasets:
    #     print('processing '+dataset[1])
    #     P = preprocesser(dataset[0],dataset[1])
    #     P.num_neg = 20
    #     P.get_train_file()
   
    
    # datasets = [['../data/train/LINE_with_attr.txt','../emb/LINE.pkl'],
    # ['../data/train/node2vec_with_attr.txt','../emb/node2vec.txt'],
    # ['../data/train/HIN2vec_with_attr.txt','../emb/HIN2vec/node.txt'],
    # ['../data/train/metapath2vec_with_attr.txt','../emb/Metapath2vec/covid-plp.txt'],
    # ['../data/train/HeGANdis_with_attr.txt','../emb/HeGAN/covid_dis.emb'],
    # ['../data/train/HeGANgen_with_attr.txt','../emb/HeGAN/covid_gen.emb'],
    # ['../data/train/HeGANmean_with_attr.txt','../emb/HeGAN/covid_mean.emb']]
    
    # # P = preprocesser('../data/train/node2vec.txt','../data/node2vec.txt')
    # for dataset in [datasets[3]]:
    # # for dataset in datasets:
    #     print('processing '+dataset[1])
    #     P = preprocesser(dataset[0],dataset[1])
    #     P.num_neg = 20
    #     print('num_neg {}'.format(P.num_neg))
    #     P.load_attr()
    #     P.get_train_file_with_attr()
        # P.get_predict_file_with_attr()
        # P.get_val

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
    #     P.num_neg = 20
    #     P.load_attr()
    #     P.get_train_file_attr_only()


    # datasets = [['../data/train/LINE_with_attr.txt','../emb/LINE.pkl'],
    # ['../data/train/node2vec_with_attr.txt','../emb/node2vec.txt'],
    # ['../data/train/HIN2vec_with_attr.txt','../emb/HIN2vec/node.txt'],
    # ['../data/train/metapath2vec_baseline.txt','../emb/Metapath2vec/covid-plp.txt'],
    # ['../data/train/HeGANdis_with_attr.txt','../emb/HeGAN/covid_dis.emb'],
    # ['../data/train/HeGANgen_with_attr.txt','../emb/HeGAN/covid_gen.emb'],
    # ['../data/train/HeGANmean_with_attr.txt','../emb/HeGAN/covid_mean.emb']]
    
    # # P = preprocesser('../data/train/node2vec.txt','../data/node2vec.txt')
    # for dataset in [datasets[3]]:
    # # for dataset in datasets:
    #     print('processing '+dataset[1])
    #     P = preprocesser(dataset[0],dataset[1])
    #     P.num_neg = 5
    #     print('num_neg {}'.format(P.num_neg))
    #     P.load_attr()
    #     P.get_baseline_train_file()




    # make local cases prediction file 
    P = preprocesser('../data/train/metapath2vec_topredict.txt','../emb/Metapath2vec/covid-plp.txt')
    P.load_attr()
    P.get_predict_file_with_attr()