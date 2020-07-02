import numpy as np

class FeatureMaker:
    def __init__(self,with_emb=True,emb_only=False):
        # self.emb_file = 'emb.txt'
        self.p2l_file = '../data/pat2loc_3.txt'
        self.out_file = '../train/feature.txt'
        self.link_file = '../data/doublelink.txt'
        self.train_file = '../train/train.txt'
        self.attr_file = '../data/attrs_hospital.txt'
        self.emb_only = emb_only
        self.with_emb = with_emb
        
    def build_feature(self):
        vocab = self.build_vocab()
        id_map,emb = self.get_emb()
        no_emb = 0
        

        numeric = ['age','touch']
        date = ['time','sick','hospital','confirm','arrive']
        onehot = ['level']
        hashmp = {}
        if not self.with_emb:
            self.out_file = '../train/feature_attr_only.txt'
            self.train_file = '../train/train_attr_only.txt'
        if  self.emb_only:
            self.out_file = '../train/feature_emb_only.txt'
            self.train_file = '../train/train_emb_only.txt'
        fout = open(self.out_file ,'w',encoding='utf-8')
        fin = open(self.attr_file ,'r',encoding='utf-8')
        lines = fin.readlines()
        fin.close()

        for line in lines:
            line = line.strip().split(' ')
            tmp = [ 0 for i in range(7+len(vocab))]
            # tmp[0] = line[0]
            # hashmp[0] = 'id'
            # tmp[0] = '0'
        #     print(tmp[0])
            for attr in line[1:]:
                att =attr.replace("亳",'毫')
                attr = attr.replace(" ","")
                attr = attr.split('_')
                if attr[0] in onehot:
        #             print(vocab[attr[-1]],len(tmp))
                    hashmp[vocab[att]] = attr[0]
                    tmp[vocab[att]] = 1          #ont-hot encode
                elif attr[0] in numeric:
                    idx = numeric.index(attr[0])
                    tmp[idx] = self.get_number(attr[-1])
                    hashmp[idx] = attr[0]
                elif attr[0] in date:
                    
                    idx = date.index(attr[0])
                    hashmp[idx+len(numeric)] = attr[0]
                    tmp[idx+len(numeric)] = self.get_time(attr[-1])
    #             else:
    #                 if attr[0] != 'location':
    #                     print("err!")
    #                     print(attr)
            
            
            if self.emb_only  and not self.with_emb:
                print('wrong parameter!')
                return 0
            
            
            # print(len(tmp))
            if not self.emb_only:
                for i,attr in enumerate(tmp):
                    fout.write(str(attr)+' ')
            if self.with_emb:
                id = 'p' + line[0]
                # print(id)
                try:
                    embeddings = emb[id_map[id]]
                except:
                    no_emb += 1
                    embeddings = [0 for _ in range(32)]
                for number in embeddings:
                    fout.write(str(number)+' ')
            fout.write('\n')
        
        print("no embdding patients: %d"%no_emb)
        return hashmp
        fout.close()

    def build_vocab(self):
        onehot = ['level']
        fin = open(self.attr_file,'r',encoding='utf-8')
        lines = fin.readlines()
        fin.close()
        vocab = {}
        count = 7    #前8个attr不是 one-hot encode
        for line in lines:
        #     print(line)
            line = line.strip().split(' ')
            for attr in line[1:]:
                att = attr.replace("亳",'毫')
                attr = attr.replace(" ","")
                attr = attr.split('_')

        #         print(attr[1])
                if attr[0] in onehot:
                    if att not in vocab:
                        vocab[att] = count
                        count += 1
        return vocab
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
    
    # 将日期转化为数字
    def get_time(self,time):
        try:
            idx1 = time.index('月')
            idx2 = time.index('日')
            return int((int(time[idx1-1])-1)*31+int(time[(idx1+1):(idx2)]))
        except:
            return 0

    #年龄等转化为数字
    def get_number(self,string):
        string = string.replace("岁","").replace(" ","")
        try:
            return int(string)
        except:
            return 0

    def get_emb(self):
        import numpy as np
        import json
        index2nodeid = json.load(open("metapath2vec/log/index2nodeid.json"))
        index2nodeid = {int(k):v for k,v in index2nodeid.items()}
        nodeid2index = {v:int(k) for k,v in index2nodeid.items()}
        node_embeddings = np.load("metapath2vec/log/node_embeddings.npz")['arr_0']

        #node embeddings of "yi"
        # node_embeddings[nodeid2index["yi"]]

        return nodeid2index,node_embeddings
    
    def if_meet(self):
        f = open(self.p2l_file,'r',encoding='utf-8')
        p2l = {}
        for line in f :
            line = line.strip().split(' ')
            if int(line[0]) not in p2l:
                p2l[int(line[0])] = [line[1]]
            else:
                p2l[int(line[0])].append(line[1])
        # print(p2l)
        f.close()
        return p2l
    
    def make_train(self,num_neg):
        
        f = open(self.out_file,'r',encoding='utf-8')
        features = []
        for line in f:
            line = line.strip().split(' ')
            # features.append([int(line[0])]+list(map(lambda x: float(x),line[1:])))
            features.append(list(map(lambda x: float(x),line)))
        print(len(features[0]))
        # print(len(features))
        f.close()
        

        f = open(self.link_file,'r',encoding='utf-8')
        positive = [] 
        for line in f:
            line = line.strip().split(' ')
            positive.append([int(line[0]),int(line[1])])
        f.close()

        f = open(self.train_file,'w',encoding='utf-8')
        
        
        p2l = self.if_meet()
        for pairs in positive:
            tmp = []
            # print(pairs,le)
            f1 = features[pairs[0]-1]
            f2 = features[pairs[1]-1]
            
            try :
                meets = len(set(p2l[pairs[0]]) & set(p2l[pairs[1]]))  #公共location 数目
            except:
                meets = 0
            tmp = f1 + f2 + [meets] + [1]   # label = 1
            tmp = list(map(lambda x:str(x),tmp))
            # print(len(tmp))
            f.write(" ".join(tmp)+'\n')
            count = 0
            while count < num_neg:
                idx = np.random.randint(1,990)
                # print(idx,len(features))
                if [pairs[0],idx] not in positive:
                    f3 = features[idx-1]
                    count += 1
                else:
                    continue
                try:
                    meets = len(set(p2l[pairs[0]]) & set(p2l[idx])) 
                    if meets > 0:
                        print(meets)
                except:
                    meets = 0
                if meets <= 2:
                    tmp = f1 + f3 + [meets] + [0]   #label =0
                    tmp = list(map(lambda x:str(x),tmp))
                    f.write(" ".join(tmp)+'\n')
        f.close()

    def make_train_2(self,num_neg):
        pos_sim = []
        neg_sim = []
        f = open(self.out_file,'r',encoding='utf-8')
        features = []
        for line in f:
            line = line.strip().split(' ')
            # features.append([int(line[0])]+list(map(lambda x: float(x),line[1:])))
            features.append(list(map(lambda x: float(x),line)))
        print(len(features[0]))
        # print(len(features))
        f.close()
        

        f = open(self.link_file,'r',encoding='utf-8')
        positive = [] 
        for line in f:
            line = line.strip().split(' ')
            positive.append([int(line[0]),int(line[1])])
        f.close()

        f = open(self.train_file,'w',encoding='utf-8')
        
        
        p2l = self.if_meet()
        for pairs in positive:
            tmp = []
            # print(pairs,le)
            f1 = features[pairs[0]-1]
            # print(len(f1))
            f2 = features[pairs[1]-1]
            
            emb_1 = f1[-32:]
            emb_2 = f2[-32:]
            emb_sim = self.cosine_similarity(emb_1,emb_2)
            try :
                meets = len(set(p2l[pairs[0]]) & set(p2l[pairs[1]]))  #公共location 数目
            except:
            # print(Exception)
                meets = 0
            if self.with_emb:
                pos_sim.append(emb_sim)
                if self.emb_only:
                    tmp =  [meets] + [emb_sim] + [1]
                else:
                    tmp = f1[:-32] + f2[:-32] + [meets] + [emb_sim] + [1]   # label = 1
            else:
                tmp = f1 + f2 + [meets] + [1]   # label = 1
            tmp = list(map(lambda x:str(x),tmp))
            # print(len(tmp))
            f.write(" ".join(tmp)+'\n')
            count = 0
            while count < num_neg:
                idx = np.random.randint(1,990)
                # print(idx,len(features))
                if [pairs[0],idx] not in positive:
                    f3 = features[idx-1]
                    count += 1
                else:
                    continue
                try:
                    meets = len(set(p2l[pairs[0]]) & set(p2l[idx])) 
                    if meets > 0:
                        print(meets)
                except:
                    meets = 0
                if meets <= 2:
                    emb_3 = f3[-32:]
                    emb_sim = self.cosine_similarity(emb_1,emb_3)
                    if self.with_emb:
                        neg_sim.append(emb_sim)
                        if self.emb_only:
                            tmp =  [meets] + [emb_sim] + [0]
                        else:
                            tmp = f1[:-32] + f2[:-32] + [meets] + [emb_sim] + [0]   #label =0
                    else:
                        tmp = f1 + f2 + [meets] + [0]   # label = 1
                    tmp = list(map(lambda x:str(x),tmp))
                    f.write(" ".join(tmp)+'\n')
        if self.with_emb:
            print(np.mean(pos_sim),np.mean(neg_sim))
            print((np.mean(pos_sim)+np.mean(neg_sim))/2)
        f.close()
                




if __name__ =='__main__':
    f = FeatureMaker(with_emb=True,emb_only=True)
    hashmp = f.build_feature()
    f.make_train_2(1)
    f = FeatureMaker(with_emb=True,emb_only=False)
    hashmp = f.build_feature()
    f.make_train_2(1)
    f = FeatureMaker(with_emb=False,emb_only=False)
    hashmp = f.build_feature()
    f.make_train_2(1)


    '''
    TODO

    1. add_emb

    2. if_meet_count
    
    
    '''
    
