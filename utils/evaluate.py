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
        self.patient_num = 1017
        self.location_num = 2388
        

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

    def load_embedding(self):
        if 'LINE' in self.emb_file:
            import pickle
            f = open(self.emb_file,'rb')
            self.emb = pickle.load(f)
            # print(self.emb[1],len(self.emb[1]))
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

    def  top_sim(self,topk=10):
        fout = open('../res/top_sim_top{}.txt'.format(topk),'a',encoding='utf-8')
        self.load_embedding()
        maxs = np.array([[0,0,0]for _ in range(topk)])
        # maxs = []
        for i in tqdm(range(1,self.patient_num+1)):
            for j in range(i+1,self.patient_num+1):
                if i == j:
                    continue
                if i not in self.emb.keys() or j not in self.emb.keys():
                    continue
                cossim = self.cosine_similarity(self.emb[i],self.emb[j])
                if cossim > min(maxs[:,:1]):
                    maxs = np.delete(maxs ,np.argmin(maxs [:,:1]),axis=0)
                    maxs = np.concatenate((maxs,[[cossim,i,j]]))
        
        maxs = sorted(maxs,key=lambda x: x[0],reverse=True)
        print(maxs)
        fout.write(self.emb_file+'\n')
        for i in range(len(maxs)):
            fout.write(str(maxs[i][1])+' '+str(maxs[i][2])+'\n')
        fout.write('*********************************\n')
        fout.close()
        


class predictor:
    def __init__(self,train_file):
        self.train_file = train_file

    def load_data(self,test_size=0.5):
        from sklearn.model_selection import train_test_split
        train_x = []
        train_y = []
        f = open(self.train_file,'r',encoding='utf-8')
        for line in f:
            line = line.strip().split(' ')
            line = list(map(lambda x: float(x),line))
            train_x.append(line[:-1])          
            train_y.append(int(line[-1]))     ##label

        x_train,x_test,y_train,y_test = train_test_split(train_x,train_y,test_size=test_size,stratify=train_y,random_state=233,shuffle=True)
        
        
        return x_train,x_test,y_train,y_test
    
    

    def score(self,y_true,pred):
        from sklearn.metrics import precision_score, recall_score, accuracy_score
        from sklearn.metrics import  roc_auc_score 

        train_y = y_true
        auc= roc_auc_score(train_y,pred) 
        # for i in range(len(pred)):
            # print(train_y[i],pred[i])
        print("auc: %s"% auc)
        return auc
    
    def LR_train(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        # train_x,train_y = self.load_data()
        times = 50
        acc = 0
        f = open('../res/acc.txt','a',encoding='utf-8')
        
        for _ in range(times):
            train_x,test_x,train_y,test_y = self.load_data()
            lr= LogisticRegression()
            lr.fit(train_x,train_y)
            pred = lr.predict(test_x)
            preds = []
            for p in pred:
                if p>0.5:
                    preds.append(1)
                else:
                    preds.append(0)
            
            acc += accuracy_score(test_y,preds)
            

        print(self.train_file)   
        print('avg acc: {}'.format(acc/times))

        f.write(self.train_file+'\n')   
        f.write('avg acc: {}\n'.format(acc/times))
        f.close()

if __name__ == '__main__':
    embs = ['../emb/LINE.pkl','../emb/node2vec.txt','../emb/HIN2vec/node.txt','../emb/Metapath2vec/covid-plp.txt','../emb/HeGAN/covid_dis.emb','../emb/HeGAN/covid_gen.emb']
    for emb in embs:
        print('processing '+emb)
        E = evaluater(emb)
        # E.compare_similarity()
        E.top_sim()



    # datasets = ['../data/train/node2vec.txt','../data/train/LINE.txt','../data/train/metapath2vec.txt',
    # '../data/train/HIN2vec.txt','../data/train/HeGANdis.txt','../data/train/HeGANgen.txt']
    # for dataset in datasets:
    #     print('processing '+dataset)
    #     P = predictor(dataset)
    #     P.LR_train()

        

