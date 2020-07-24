import numpy as np

class FeatureMaker:
    def __init__(self,with_emb=True,emb_only=False,t_th=3,d_th=3):
        # self.emb_file = 'emb.txt'
        self.p2l_file = '../data/p2l.txt'
        self.out_file = '../data/train/feature_hk.txt'
        self.link_file = '../data/doublelink_hk.txt'
        self.train_file = '../data/train/train_hk.txt'
        self.attr_file = '../data/preprocess/attrs_hk_raw.txt'
        self.emb_only = emb_only
        self.with_emb = with_emb
        self.t_th = t_th
        self.d_th = d_th
        self.pairs_file = '../data/pairs.txt'

    
    
    def check_attr(self,idx):
        f = open(self.link_file,'r',encoding='utf-8')
        pos = []
        for line in f:
            line = line.split(' ')
            if [int(line[0]),int(line[1])] in pos:
                print(line)
            pos.append([int(line[0]),int(line[1])]) 
        f.close()

        attrs = []
        f = open('../train/feature_attr_only_hk.txt','r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            line = list(map(lambda x: float(x),line))
            attrs.append(line)
        f.close()
        postives = []
        negatives = []

        for pair in pos:
            attr1 = attrs[pair[0]-1][idx]
            attr2 = attrs[pair[1]-1][idx]
            postives.append(abs(attr1-attr2))
        
        for i in range(1017):
            for j in range(1017):
                if [i+1,j+1] not in pos:
                    attr1 = attrs[i][idx]
                    attr2 = attrs[j][idx]
                    negatives.append(abs(attr1-attr2))
        print('pos avg abs: {}'.format(np.mean(postives)))
        print('neg avg abs: {}'.format(np.mean(negatives)))

        
        


    def make_hash(self,vocab):
        attrs = ['age','onset','confirmation']
        vocab_hash = {0:0,1:1,2:2} 
        for key,value in vocab.items():
            key = key.split('_')[0]
            if key not in attrs:
                attrs.append(key)
            vocab_hash[value] = attrs.index(key)
        print(vocab_hash)
            
         
        return 0
    def build_feature(self):
        # id_map,emb = self.get_emb()
        no_emb = 0
        

        numeric = ['age']
        date = ['onset','confirmation']
        # onehot = ['hospital_zh,','status','type','classification','citizenship''gender','location']
        # onehot = ['hospital','status','type','classification','citizenship','gender']
        # onehot = ['status','type','classification','gender']
        onehot = ['classification','citizenship','hospital']
        vocab = self.build_vocab(len(numeric)+len(date))
        self.make_hash(vocab)
        # print(vocab)

        # print(vocab)
        # print(len(vocab))
        # onehot = ['status','gender','place']
       
        if not self.with_emb:
            self.out_file = '../data/train/feature_attr_only_hk.txt'
            self.train_file = '../train/train_attr_only_hk_{}_{}.txt'.format(self.t_th,self.d_th)
        if  self.emb_only:
            self.out_file = '../train/feature_emb_only_hk.txt'
            self.train_file = '../train/train_emb_only_hk_{}_{}.txt'.format(self.t_th,self.d_th)
        if not self.emb_only and self.with_emb:
            self.train_file = '../train/train_hk_{}_{}.txt'.format(self.t_th,self.d_th)
        fout = open(self.out_file ,'w',encoding='utf-8')
        fin = open(self.attr_file ,'r',encoding='utf-8')
        lines = fin.readlines()
        fin.close()

        for line in lines:
            line = line.strip().split(' ')
            tmp = [ 0 for i in range(len(numeric)+len(date)+len(vocab))]
            # tmp[0] = line[0]
            # hashmp[0] = 'id'
            # tmp[0] = '0'
        #     print(tmp[0])
            for attr in line[1:]:
                att =attr.replace("亳",'毫')
                attr = attr.replace(" ","")
                attr = attr.split('_')
                if attr[0] in onehot:
                    if attr[0] == 'status' and attr[1] == 'zh':
                        continue
                    if attr[0] == 'classification' and attr[1] == 'zh':
                        continue
                    if attr[0] == 'group' and attr[1] == 'name':
                        continue
                    if attr[0] == 'group' and attr[1] == 'related':
                        continue
        #             print(vocab[attr[-1]],len(tmp))
                    tmp[vocab[att]] = 1          #ont-hot encode
                elif attr[0] in numeric:
                    idx = numeric.index(attr[0])
                    tmp[idx] = self.get_number(attr[-1])   
                elif attr[0] in date:
                    idx = date.index(attr[0]) 
                    tmp[idx+len(numeric)] = self.get_date(attr[-1])
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
                id = 'p' + line[0].split('_')[-1]
                # print(id)
                try:
                    # embeddings = emb[id_map[id]]
                    embedding = 0
                except:
                    no_emb += 1
                    embeddings = [0 for _ in range(32)]
                for number in embeddings:
                    fout.write(str(number)+' ')
            fout.write('\n')
        
        print("no embdding patients: %d"%no_emb)

        fout.close()

    def build_vocab(self,count):
        # onehot = ['hospital_zh,','status','type','classification','citizenship''gender','location']
        # onehot = ['hospital','status','type','classification','citizenship','gender']
        # onehot = ['status','type','classification','gender']
        onehot = ['classification','citizenship','hospital']
        # onehot = ['status','gender','place']
        fin = open(self.attr_file,'r',encoding='utf-8')
        lines = fin.readlines()
        fin.close()
        vocab = {}
        # count = 2    #前8个attr不是 one-hot encode
        for line in lines:
        #     print(line)
            line = line.strip().split(' ')
            for attr in line[1:]:
                att = attr.replace("亳",'毫')
                attr = attr.replace(" ","")
                attr = attr.split('_')

        #         print(attr[1])
                if attr[0] in onehot:
                    if attr[0] == 'status' and attr[1] == 'zh':
                        continue
                    if attr[0] == 'classification' and attr[1] == 'zh':
                        continue
                    if attr[0] == 'group' and attr[1] == 'name':
                        continue
                    if attr[0] == 'group' and attr[1] == 'related':
                        continue
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
    
    def get_date(self,date):
        if '-' not in date:
            return 0            #无日期 按0 处理
        date = date.split('-')
        month = date[1]
        day = date[2]
        if month == '01':
            month = 0
        elif month == '02':
            month = 31
        elif month == '03':
            month = 60
        else:
            month = 91
        if day[0] == '0':
            day = day[1]
        return month + int(day)

    # 将日期转化为数字
    def get_time(self,time):
        try:
            idx1 = time.index('月')
            idx2 = time.index('日')
            return int((int(time[idx1-1])-1)*30+int(time[(idx1+1):(idx2)]))
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
        index2nodeid = json.load(open("metapath2vec/log_{}{}/index2nodeid.json".format(self.t_th,self.d_th)))
        index2nodeid = {int(k):v for k,v in index2nodeid.items()}
        nodeid2index = {v:int(k) for k,v in index2nodeid.items()}
        node_embeddings = np.load("metapath2vec/log_{}{}/node_embeddings.npz".format(self.t_th,self.d_th))['arr_0']

        # print(nodeid2index)
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
        print('feature length:{}'.format(len(features[0])))
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
            f2 = features[pairs[0]-1]
            
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
    def make_pairs(self,num_neg):
        pos_sim = []
        neg_sim = []
        fp = open(self.pairs_file,'w',encoding='utf-8')
        # f = open('../train/feature_old.txt','r',encoding='utf-8')
        f = open(self.out_file,'r',encoding='utf-8')
        features = []
        for line in f:
            line = line.strip().split(' ')
            # features.append([int(line[0])]+list(map(lambda x: float(x),line[1:])))
            features.append(list(map(lambda x: float(x),line)))
        print('feature length:{}'.format(len(features[0])))
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
                # tmp = f1[:-32] + f2[:-32] + [meets] + [1]   # label = 1
            tmp = list(map(lambda x:str(x),tmp))
            # print(len(tmp))
            f.write(" ".join(tmp)+'\n')
            fp.write(str(pairs[0])+' '+str(pairs[1])+'\n')
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
                            tmp = f1[:-32] + f3[:-32] + [meets] + [emb_sim] + [0]   #label =0
                    else:
                        tmp = f1 + f3 + [meets] + [0]   # label = 1
                        # tmp = f1[:-32] + f3[:-32] + [meets] + [0]   # label = 1
                    tmp = list(map(lambda x:str(x),tmp))
                    f.write(" ".join(tmp)+'\n')
                    fp.write(str(pairs[0])+' '+str(idx)+'\n')
                else:
                    count -= 1
        if self.with_emb:
            print(np.mean(pos_sim),np.mean(neg_sim))
            print((np.mean(pos_sim)+np.mean(neg_sim))/2)
        f.close()
        fp.close()
    def make_train_2(self):
        pos_sim = []
        neg_sim = []
        # self.make_pairs()
        
        # f = open('../train/feature_old.txt','r',encoding='utf-8')
        f = open(self.out_file,'r',encoding='utf-8')
        features = []
        for line in f:
            line = line.strip().split(' ')
            # features.append([int(line[0])]+list(map(lambda x: float(x),line[1:])))
            features.append(list(map(lambda x: float(x),line)))
        print('featute:{} '.format(len(features[0])))
        # print(len(features))
        f.close()
        

        f = open(self.pairs_file,'r',encoding='utf-8')
        pairss = [] 
        for line in f:
            line = line.strip().split(' ')
            pairss.append([int(line[0]),int(line[1])])
        f.close()

        f = open(self.train_file,'w',encoding='utf-8')
        
        
        p2l = self.if_meet()
        for i,pairs in enumerate(pairss):
            if i % 2 == 0:
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
                    # tmp = f1[:-32] + f2[:-32] + [meets] + [1]   # label = 1
                tmp = list(map(lambda x:str(x),tmp))
                # print(len(tmp))
                f.write(" ".join(tmp)+'\n')
            
            
            else:
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
                    neg_sim.append(emb_sim)
                    if self.emb_only:
                        tmp =  [meets] + [emb_sim] + [0]
                    else:
                        tmp = f1[:-32] + f2[:-32] + [meets] + [emb_sim] + [0]   # label = 1
                else:
                    tmp = f1 + f2 + [meets] + [0]   # label = 1
                    # tmp = f1[:-32] + f2[:-32] + [meets] + [1]   # label = 1
                tmp = list(map(lambda x:str(x),tmp))
                # print(len(tmp))
                f.write(" ".join(tmp)+'\n')
        if self.with_emb:
            print(np.mean(pos_sim),np.mean(neg_sim))
            print((np.mean(pos_sim)+np.mean(neg_sim))/2)
        f.close()

    def check_local(self):
        locals = []
        f = open('../../hk/data/local.txt','r',encoding='utf-8')
        for line in f:
            line = line.strip()
            locals.append(int(line))
        f.close()

        attrs = []
        f = open('../train/feature_attr_only_hk.txt','r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            line = list(map(lambda x: float(x),line))
            attrs.append(line)
        f.close()

        id_map,emb = self.get_emb()

        res = {}
        # from sklearn.externals import joblib
        # clf=joblib.load('rf_hk.pkl')

        p2l = self.if_meet()
        print(locals)
        # # 基于地点的查找
        # for local in locals:
        #     tmp = []
        #     for idx in range(1,1018):
        #         if idx == local:
        #             continue
        #         # elif idx not in locals:
        #         #     continue
        #         else:
        #             try:
        #                 # sim = len(set(p2l[local]) & set(p2l[idx]))
        #                 sim = self.cosine_similarity(emb[id_map['p'+str(local)]],emb[id_map['p'+str(idx)]])
        #             except:
        #                 sim = 0
        #             tmp.append([sim,idx])
        #     tmp = sorted(tmp,key=lambda x:x[0], reverse=True)
        #     res[local] = tmp[:10]
        # print(res[857])
        # print(res.keys())

        # temp = []
        # for key,values in res.items():
        #     for value in values:
        #         temp.append([value[0],key,value[1]])
        # temp = sorted(temp,key=lambda x:x[0], reverse=True)
        # print(temp[:20])


        # res2 = {}
        # # 基于属性的
        for local in locals:
        # for local in [16]:
            train = []
            for idx in range(1017):
                id = idx + 1
                if id == local:
                    continue
                else:
                    try:
                         meets = len(set(p2l[local]) & set(p2l[id]))
                    except:
                        meets = 0
                    tmp = attrs[local-1] + attrs[id-1] + [meets]
                    train.append(tmp)
        # print(train)
            x = clf.predict(train)
        # print(x.index())
        # x = sorted(x,reverse=True)
        # print(x[:100])
        # print(len(x))
            # print(local)
            index = np.argwhere(x==1.0).tolist()
            index1 = np.argwhere(x==0.9).tolist()
            index2 = np.argwhere(x==0.8).tolist()
            index2 = np.argwhere(x==0.7).tolist()
            index2 = np.argwhere(x==0.6).tolist()
            # res[local] = index
        
            res[local] = [index] + [index1] + [index2]
        print(res[857])
        #     # print(res2[local])
        # for key,value in res.items():
        #     print(key,value)
        # print(x.index(1.0))
        # print(np.mean(x),np.max(x))
        # for key in res.keys():
        #     idxs = []
        #     all = res[key]
        #     for x in all:
        #         idxs.append(x[1])
        #     idxs2 = res2[key]
        #     # print(idxs,idxs2)
        #     common = [val for val in idxs if val in idxs2]
        #     # common = list(set(idxs).intersection(set(idxs2)))
        #     if len(common) > 0:
        #         print(key,common)





    def make_train_3(self,num_neg):
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
        
        f = open('../../hk/data/hospital_hk.txt','r',encoding='utf-8')
        hospitals = {}
        count = 1  
        for line in f:
            line = line.strip()
            hospitals[count] = line
            count += 1
        f.close()
        # print(hospitals)

        f = open('../../hk/data/citizenship_hk.txt','r',encoding='utf-8')
        citizenships = {}
        count = 1  
        for line in f:
            line = line.strip()
            citizenships[count] = line
            count += 1
        f.close()

        f = open(self.link_file,'r',encoding='utf-8')
        positive = [] 
        for line in f:
            line = line.strip().split(' ')
            positive.append([int(line[0]),int(line[1])])
        f.close()

        f = open(self.train_file,'w',encoding='utf-8')
        
        meets_total = 0
        meets_total_neg = 0
        meets_count = 0
        hos_count = 0
        hos_count_neg = 0
        city_count = 0
        city_count_neg = 0
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

            if citizenships[pairs[0]] == citizenships[pairs[1]]:
                citizenship = 1
                city_count += 1
            else:
                citizenship = 0
                city_count_neg +=1

            if hospitals[pairs[0]] == hospitals[pairs[1]]:
                hospital = 1
                hos_count += 1
            else:
                hospital = 0
                hos_count_neg += 1
            try :
                meets = len(set(p2l[pairs[0]]) & set(p2l[pairs[1]]))  #公共location 数目
            except:
            # print(Exception)
                meets = 0
            meets_total += meets
            if meets > 0:
                meets_count += 1
            if self.with_emb:
                pos_sim.append(emb_sim)
                if self.emb_only:
                    tmp =  [meets] + [emb_sim] + [1]
                else:
                    tmp = f1[:-32] + f2[:-32] + [meets] + [emb_sim] + [hospital] + [citizenship] + [1]   # label = 1
            else:
                tmp = f1 + f2 + [meets] +  [hospital] + [citizenship] + [1]   # label = 1
            tmp = list(map(lambda x:str(x),tmp))
            # print(len(tmp))
            f.write(" ".join(tmp)+'\n')
            count = 0
            while count < num_neg:
                idx = np.random.randint(1,412)
                # print(idx,len(features))
                if [pairs[0],idx] not in positive:
                    f3 = features[idx-1]
                    count += 1
                else:
                    continue
                try:
                    meets = len(set(p2l[pairs[0]]) & set(p2l[idx])) 
                    if meets > 0:
                        # print(meets)
                        pass
                except:
                    meets = 0
                if meets <= 1:
                    meets_total_neg += meets
                    emb_3 = f3[-32:]
                    emb_sim = self.cosine_similarity(emb_1,emb_3)
                    if self.with_emb:
                        neg_sim.append(emb_sim)
                        if self.emb_only:
                            tmp =  [meets] + [emb_sim] + [0]
                        else:
                            tmp = f1[:-32] + f3[:-32] + [meets] + [emb_sim] + [hospital] + [citizenship] + [0]   #label =0
                    else:
                        tmp = f1 + f3 + [meets] + [hospital] + [citizenship] + [0]   # label = 1
                    tmp = list(map(lambda x:str(x),tmp))
                    f.write(" ".join(tmp)+'\n')
        if self.with_emb:
            print(np.mean(pos_sim),np.mean(neg_sim))
            print((np.mean(pos_sim)+np.mean(neg_sim))/2)
        print(meets_total,meets_total_neg,meets_total/len(positive))
        print(len(positive),meets_count)
        print(len(positive),hos_count,hos_count_neg,city_count,city_count_neg)
        f.close()
                




if __name__ =='__main__':
    # emb only
    # f = FeatureMaker(with_emb=True,emb_only=True)
    # f.build_feature()
    # f.make_train_2(1)

    # # attr + emb
    # f = FeatureMaker(with_emb=True,emb_only=False)
    # f.build_feature()
    # f.make_train_2(1)

    # # attr only
    f = FeatureMaker(with_emb=False,emb_only=False)
    f.build_feature()
    # f.make_train_2(1)
    # f = FeatureMaker(with_emb=True,emb_only=True)
    # f.check_attr(0)
    # f.check_attr(1)
    # f.check_attr(2)
    # f.check_local()
    # f.make_pairs(1)
    # # for sx,sy in [[1,1],[2,2],[3,3],[10,10]]:
    # for sx,sy in [[3,3]]:
    #     f = FeatureMaker(with_emb=True,emb_only=True,t_th=sx,d_th=sy)
    #     f.build_feature()
    #     f.make_train_2()

    #     # attr + emb
    #     f = FeatureMaker(with_emb=True,emb_only=False,t_th=sx,d_th=sy)
    #     f.build_feature()
    #     f.make_train_2()

    #     # attr only
    #     f = FeatureMaker(with_emb=False,emb_only=False,t_th=sx,d_th=sy)
    #     f.build_feature()
    #     f.make_train_2()
    



    '''
    TODO

    1. add_emb

    2. if_meet_count
    
    
    '''
    
