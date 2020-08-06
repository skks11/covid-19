import numpy as np
from tqdm import tqdm

class evaluater:
    def __init__(self,emb_file):
        self.l2l = '../data/l2l.txt'
        self.p2l = '../data/p2l.txt'
        self.ptol = {}
        self.doublelink = '../data/doublelink_hk.txt'
        self.emb_file = emb_file
        self.emb = {}
        self.positive = []
        self.negtive = []
        self.patient_num = 2000
        self.location_num = 8233
        self.attr = {}
        

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

    ##load属性信息并归一化
    def load_attr(self):
        f = open('../data/feature_attr_only_hk.txt','r',encoding='utf-8')
        tmp = []
        for line in f:
            line = line.strip().split()
            attrs  = list(map(lambda x: int(x),line))
            tmp.append(attrs)
        f.close()
        tmp = np.asarray(tmp)
        tmp = tmp / tmp.max(axis=0)
        # print(tmp[:2])
        for i in range(len(tmp)):
            self.attr[i+1] = tmp[i]
        # print(self.attr[1])
    
    def top_attr_sim(self,topk=10):
        fout = open('../res/top_attr_sim_top.txt','a',encoding='utf-8')
        maxs = np.array([[0,0,0]for _ in range(topk)])
        # maxs = []
        self.get_pos()
        for i in tqdm(range(1,self.patient_num+1)):
            for j in range(i+1,self.patient_num+1):
                if i == j:
                    continue
                cossim = self.cosine_similarity(self.attr[i],self.attr[j])
                if cossim > min(maxs[:,:1]):
                    maxs = np.delete(maxs ,np.argmin(maxs [:,:1]),axis=0)
                    maxs = np.concatenate((maxs,[[cossim,i,j]]))
        # fout.write(self.emb_file+'\n')
        maxs = sorted(maxs,key=lambda x: x[0],reverse=True)
        for i in range(len(maxs)):
            fout.write(str(maxs[i][1])+' '+str(maxs[i][2]))
            if [int(maxs[i][1]),int(maxs[i][2])] in self.positive:
                fout.write(' yes')
            else:
                fout.write(' no')
            fout.write('\n')
        fout.write('*********************************\n')
        fout.close()
    # 随机生成负例
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
        self.get_pos()
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
        # print(maxs)
        fout.write(self.emb_file+'\n')
        for i in range(len(maxs)):
            fout.write(str(maxs[i][1])+' '+str(maxs[i][2]))
            # print(maxs[i])
            if [int(maxs[i][1]),int(maxs[i][2])] in self.positive:
                fout.write(' yes')
            else:
                fout.write(' no')
            fout.write('\n')
        fout.write('*********************************\n')
        fout.close()
        


class predictor:
    def __init__(self,train_file):
        self.train_file = train_file

    def load_data(self,baseline=False,test_size=0.5):
        from sklearn.model_selection import train_test_split
        train_x = []
        train_y = []

        if baseline:
            input_file = '../data/train/metapath2vec_baseline.txt'
        else:
            input_file = self.train_file
        
        f = open(input_file,'r',encoding='utf-8')
        for line in f:
            line = line.strip().split(' ')
            line = list(map(lambda x: float(x),line))
            train_x.append(line[:-1])          
            train_y.append(int(line[-1]))     ##label
        f.close()

        

        x_train,x_test,y_train,y_test = train_test_split(train_x,train_y,test_size=test_size,random_state=100,stratify=train_y,shuffle=True)
        
         
        return x_train,x_test,y_train,y_test
    
    # def plotPR(self,classifier,x_test,y_test,preds):
    #     from sklearn.metrics import precision_recall_curve
    #     # from sklearn.metrics import plot_precision_recall_curve
    #     import matplotlib.pyplot as plt
        # from sklearn.metrics import average_precision_score
        # average_precision = average_precision_score(y_test, preds)

        # disp = plot_precision_recall_curve(classifier, x_test, y_test)

        # disp.ax_.set_title(self.train_file+' '
                #    'AP={0:0.2f}'.format(0.88))

    def recall_at_topk(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score,f1_score,classification_report,precision_recall_curve
        from sklearn.model_selection import train_test_split
        
        
        pairs = []
        f = open('../data/train/mapping.txt','r',encoding='utf-8')
        for line in f:
            line= line.strip().split()
            pairs.append([int(line[0]),int(line[1])])
        f.close()

        train_x = []
        train_y = []
        
        f = open(self.train_file,'r',encoding='utf-8')
        for line in f:
            line = line.strip().split(' ')
            line = list(map(lambda x: float(x),line))
            train_x.append(line[:-1])          
            train_y.append(int(line[-1]))     ##label
        f.close()

        p_train,p_test,y_train,y_test = train_test_split(pairs,train_y,test_size=0.5,random_state=100,stratify=train_y,shuffle=True)


        # train_x,train_y = self.load_data()
        times = 1
        acc = 0
        f1 = 0
        f = open('../res/acc.txt','a',encoding='utf-8')
        
        for _ in range(times):
            train_x,test_x,train_y,test_y = self.load_data()
            lr= LogisticRegression()
            lr.fit(train_x,train_y)
            pred = lr.predict(test_x)
            prob = lr.predict_proba(test_x)
            
            
            # acc += accuracy_score(test_y,pred)
            # f1 += f1_score(test_y,pred)
        
        all_pos = np.sum(test_y)
        print('all pos '+str(all_pos))
        pos = []
        for i in range(len(pred)):
            pos.append([prob[i][1],i])
        pos = sorted(pos,key=lambda x: x[0],reverse=True)
        

        topks = [500,1000,1569]
        for topk in topks:
            cnt = 0
            for i in range(topk):
                if test_y[pos[i][1]] == 1:
                    cnt += 1
                # if test_y[pos[i][1]] == 0:
                #     print(p_test[pos[i][1]])
            print(str(cnt) + '     topk = '+str(topk))
        # print(cnt)

        print(pos[topk],pred[pos[i][1]])
        cnt =0
        # print(classification_report(test_y,pred))
        # print(np.sum(pred))
        for i in range(len(pred)):
            if pred[i] == 1 and test_y[i] == 1:
                cnt += 1
        print(cnt)
        print(cnt/1569)

        pred2 = []
        thresh = pos[topk][0]
        for i in range(len(prob)):
            if prob[i][1] >= thresh:
                pred2.append(1)
            else:
                pred2.append(0)
        
        print(classification_report(test_y,pred2))


        # f.write(self.train_file+'\n')   
        # f.write('avg acc: {}\n'.format(acc/times))
        f.close()

    

    def score(self,y_true,pred):
        from sklearn.metrics import precision_score, recall_score, accuracy_score
        from sklearn.metrics import  roc_auc_score 

        train_y = y_true
        auc= roc_auc_score(train_y,pred) 
        # for i in range(len(pred)):
            # print(train_y[i],pred[i])
        print("auc: %s"% auc)
        return auc
    
    def SVM_train(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn import svm
        from sklearn.metrics import accuracy_score,f1_score,classification_report,precision_recall_curve
        # train_x,train_y = self.load_data()
        times = 1
        acc = 0
        f1 = 0
        f = open('../res/acc.txt','a',encoding='utf-8')
        
        for _ in range(times):
            train_x,test_x,train_y,test_y = self.load_data()
            # lr= LogisticRegression(class_weight='balanced')
            clf = svm.SVC(probability=True)
            clf.fit(train_x,train_y)
            pred = clf.predict(test_x)
            
            # prob = lr.predict_proba(test_x)
            
        
            acc += accuracy_score(test_y,pred)
            f1 += f1_score(test_y,pred)
        
        
        print('avg acc: {}'.format(acc/times))
        print(classification_report(test_y,pred))

        # f.write(self.train_file+'\n')   
        # f.write('avg acc: {}\n'.format(acc/times))
        f.close()

        import pickle
        # 保存模型
        output = open("../model/svm.pkl", 'wb')
        pickle.dump(clf, output)  
        output.close()

        ############## for validation
        # valid = []
        # f = open('../data/train/valid.txt','r',encoding='utf-8')
        # for line in f:
        #     line = line.strip().split(' ')
        #     line = list(map(lambda x: float(x),line))  
        #     valid.append(line)
        # f.close()
        # valid_y = [1 for _ in range(len(valid))]
        # valid_pred = lr.predict(valid)
        # print('FOR VALIDATION')
        # print(classification_report(valid_y,valid_pred))


        ############## for predicting local cases

    def load_p2l(self):
        print('loading p2l......')
        ptol = {}
        f = open('../data/p2l.txt','r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            if int(line[0]) in ptol:
                ptol[int(line[0])].append(int(line[1]))
            else:
                ptol[int(line[0])] = [int(line[1])]
        for i in range(1,2001):
            if i not in ptol:
                ptol[i] = []
        f.close()
        return ptol

    def predict_local_cases(self):
        import pickle
        model = open("../model/svm.pkl", 'rb')
        svm = pickle.load(model)
        model.close()

        ptol = self.load_p2l()
        print(svm.classes_)
        topredict = []
        maps = []
        f = open('../data/train/metapath2vec_topredict.txt','r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            tmp = list(map(lambda x:float(x),line))
            topredict.append(tmp[:-2])
            maps.append(tmp[-2:])

        prob = svm.predict_proba(topredict)
        
        predict = []
        for i in range(len(prob)):
            predict.append([prob[i][1],maps[i][0],maps[i][1]])

        predict = sorted(predict,key=lambda x: x[0],reverse=True)
        
        topk = 1000
        for i in range(topk):
            p1 = predict[i][1]
            p2 = predict[i][2]
            common_location = len(set(ptol[p1]) & set(ptol[p2]))
            if common_location > 0 :
                print(p1,p2)
        



            
        

    def baseline(self):
        from sklearn.metrics import classification_report
        train_x,test_x,train_y,test_y = self.load_data(baseline=True)
        pred = []
        for data in test_x:
            if data[2] > 0:
                pred.append(1)    #有共同地点
                continue

            if data[3] > 14:
                pred.append(0)    #间隔超过两周
                continue

            if data[0] > 0.5:
                pred.append(1)
                continue
            
            if data[0] > 0.11 and data[3] <=3:  #emb similarity
                pred.append(1)
            elif data[1] > 0.6 :     # attr similarity
                pred.append(1)
            else:
                pred.append(0)

        print(classification_report(test_y,pred))

    def LR_train(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score,f1_score,classification_report,precision_recall_curve
        # train_x,train_y = self.load_data()
        times = 1
        acc = 0
        f1 = 0
        f = open('../res/acc.txt','a',encoding='utf-8')
        
        for _ in range(times):
            train_x,test_x,train_y,test_y = self.load_data()
            lr= LogisticRegression(class_weight='balanced')
            # lr= LogisticRegression()
            lr.fit(train_x,train_y)
            pred = lr.predict(test_x)
            prob = lr.predict_proba(test_x)
            
            
            acc += accuracy_score(test_y,pred)
            f1 += f1_score(test_y,pred)
        
        
        print('avg acc: {}'.format(acc/times))
        print(classification_report(test_y,pred))

        # f.write(self.train_file+'\n')   
        # f.write('avg acc: {}\n'.format(acc/times))
        f.close()


        ############## for validation
        valid = []
        f = open('../data/train/valid.txt','r',encoding='utf-8')
        for line in f:
            line = line.strip().split(' ')
            line = list(map(lambda x: float(x),line))  
            valid.append(line)
        f.close()
        valid_y = [1 for _ in range(len(valid))]
        valid_pred = lr.predict(valid)
        print('FOR VALIDATION')
        print(classification_report(valid_y,valid_pred))

        # 检查得分低的valid pairs的编号
        pos = []
        f = open('../data/preprocess/validset.txt','r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            if int(line[0]) > 2000 or int(line[0]) > 2000:
                continue
            pos.append([int(line[0]),int(line[1])])
        f.close()
        valid_prob = lr.predict_proba(valid)
        print(valid_prob[:5])
        
        for idx in np.argsort(valid_prob[:,1])[-5:]:
            print(valid_prob[idx])
            print(pos[idx])
            # print(valid[idx])
            
        # print(pos[np.argmin(valid_prob[:,1])])


        
if __name__ == '__main__':
    # 
    

    
    # emb_only
    # datasets = ['../data/train/node2vec.txt','../data/train/LINE.txt','../data/train/metapath2vec.txt',
    # '../data/train/HIN2vec.txt','../data/train/HeGANdis.txt','../data/train/HeGANgen.txt','../data/train/HeGANmean.txt']
    # for dataset in [datasets[-1],datasets[2]]:
    #     print('processing '+dataset)
    #     P = predictor(dataset)
    # #     P.LR_train()
    #     P.recall_at_topk()
        
    

    # emb + attr
    datasets = ['../data/train/node2vec_with_attr.txt','../data/train/LINE_with_attr.txt','../data/train/metapath2vec_with_attr.txt',
    '../data/train/HIN2vec_with_attr.txt','../data/train/HeGANdis_with_attr.txt','../data/train/HeGANgen_with_attr.txt','../data/train/HeGANmean_with_attr.txt']
    for dataset in [datasets[2]]:
    # for dataset in datasets:
        print('processing '+dataset)
        P = predictor(dataset)
        # P.LR_train()
        P.baseline()
        # P.SVM_train()
        # P.predict_local_cases()
        # P.recall_at_topk()

    

    # attr only
    # datasets = ['../data/train/node2vec_attr_only.txt','../data/train/LINE_attr_only.txt','../data/train/metapath2vec_attr_only.txt',
    # '../data/train/HIN2vec_attr_only.txt','../data/train/HeGANdis_attr_only.txt','../data/train/HeGANgen_attr_only.txt']
    # # for dataset in datasets:
    # for dataset in [datasets[2]]:
    #     print('processing '+dataset)
    #     P = predictor(dataset)
    #     # P.LR_train()
    #     P.recall_at_topk()
    

        

