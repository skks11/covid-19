import numpy as np
from geopy.distance import geodesic
import os
class CovidPreprocess:
    def __init__(self,t_th,d_th):
        self.date = ['onset_date','confirmation_date']
        self.onehot = ['hospital_zh','status_zh','type_zh','citizenship_zh','classification_zh','group_name_zh','group_id','group_related_cases']
        self.skip = ['action_交通','action_求醫']
        # self.ban = ['武漢','深圳','咸寧','英國、新加坡','南韓、美國','南非','杜拜','瑞士','老撾','菲律賓','加拿大',
        # '美國','印度孟買']
        self.l2l = {}
        self.t_th = t_th
        self.places = {}
        self.d_th = d_th
       
        
    def get_map(self,start = 0):
        import requests
        import json
        from tqdm import tqdm
        import time 
        f1 = open('../data/attrs_hk_raw.txt','a',encoding='utf-8')
        f2= open('../data/places_hk.txt','a',encoding='utf-8')
        for i in tqdm(range(start,1017)):
            time.sleep(0.5)
            url = 'https://wars.vote4.hk/page-data/cases/{}/page-data.json'.format(i+1)
            for _ in range(5):
                try:
                    r = requests.get(url)
                    break
                except:
                    print(str(i+1)+'failed')
            p = json.loads(r.text)
            attrs = p['result']['pageContext']['node']
            for key,value in attrs.items():
                key = key.replace(' ','')      
                if key in ['group_description_zh']:
                    continue
                tmp = key.split('_')
                # print(tmp)
                if tmp[-1] == 'en' or tmp[0] == 'detail' or tmp[-1] == 'url':
                    continue
                if len(tmp) == 1 and tmp[0] in ['classfication']:
                    continue
                if value == None or value=='':
                    f1.write(key+'_None'+' ')   
                else:    
                    f1.write(key+'_'+str(value)+' ')       
            f1.write('\n')

            try:
                if len(p['result']['pageContext']['patientGroup']) > 1:
                    print('more than one group for #'+str(i+1))
                places = p['result']['pageContext']['patientGroup'][0]['edges']
                for place in places:
                    place =place['node']
                    f2.write(str(i+1)+' '+'onset-date_{}  confirm-date_{} '.format(attrs['onset_date'],attrs['confirmation_date']))
                    f2.write('start-date_'+place[ 'start_date']+' ')
                    f2.write('end-date_'+place[ 'end_date']+' ')
                    f2.write('action_'+place[ 'action_zh']+' ')
                    f2.write(place['location_zh']+'\n')  
                f1.flush()
                f2.flush()
            except:
                print('no place info for #'+str(i+1))
        f1.close()
        f2.close()
    
    def get_distance(self,l1,l2):
        dis=geodesic((l1[1],l1[0]), (l2[1],l2[0])).km
        return dis

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

    def google(self,address):
        import json
        from urllib.request import urlopen, quote
        import requests
        import time
        time.sleep(0.1)
        my_key = 'AIzaSyDkbWtwkaKQsI03QqGG1tt6y8Z3SApWf8Y'
        url = 'https://maps.googleapis.com/maps/api/geocode/json?'
        
        url = url + 'address=' +address + '&region=hk&key=' + my_key
        # print(url)
        req = requests.get(url)

        temp = json.loads(req.text)
        # print(temp)
        # print(temp)
        try:
            lat = temp['results'][0]["geometry"]["location"]['lat']
            lng = temp['results'][0]["geometry"]["location"]['lng']
            if lat < 22 or lat>23 or lng<113 or lng >115:
                print('{} not in hk!'.format(address))
                return 0,0
            return (lng,lat)
        except:
            print('google failed')
            print(temp)
            return 0,0

    
            
    # def get_p2l(self):
    #     # get patient 2 loction file (patient-id location-id weight)
    #     p2l_ori = {}
    #     f = open('../data/p2l.txt','r',encoding='utf-8')
    #     for line in f:
    #         line = line.strip().split()
    #         if line[0] not in p2l_ori:
    #             p2l_ori[line[0]] = line[1]
    #     f.close()

    #     f = open('../data/places_hk_gps.txt','r',encoding='utf-8')
    #     self.places = {}
    #     for line in f:
    #         line = line.strip().split()
    #         if line[-1] =='0':
    #             continue
    #         self.places[int(line[2])] = [int(line[0]),float(line[3]),float(line[4])]
    #     f.close()

    #     f = open('../data/p2l_{}.txt'.format(self.t_th),'w',encoding='utf-8')
    #     for key,locs in p2l_ori.items():
    #         for  loc in locs:
    #             f.write(str(key)+' '+str(loc)+'\n')
    #             for key2,value in self.places.items():
    #                 dis = self.get_distance(self.places[key][1:],value[1:])
    #                 if dis <se

        

    #     return 0 

    def g_l2l(self,key1):
        value1 = self.places[key1]
        tmp = []
        # print(key1)
        
        for key2,value2 in self.places.items():
            if key1 == key2:
                continue
            dis = self.get_distance(value1[1:],value2[1:])
            if dis < self.d_th:
                time = abs(value1[0]-value2[0])
                if time < self.t_th:
                    tmp.append([key2,((self.d_th - dis)/self.d_th) * np.e**(-time)])
        tmp  =sorted(tmp,key=lambda x: x[1],reverse=True)  #从大到小
        l2l = {key1:tmp[:10]}
        return l2l
        
            
        # print(self.l2l[key1])
    
    def get_validset(self):
        import re
        f = open('../data/preprocess/doublelink_hk.txt','r',encoding='utf-8')
        pos = []
        for line in f:
            line = line.strip().split()
            pos.append([int(line[0]),int(line[1])])
        f.close()

        fout = open('../data/preprocess/validset.txt','w',encoding='utf-8')
        f = open('../data/preprocess/details.txt','r',encoding='utf-8')
        for pid,line in enumerate(f):
            pid = pid + 1
            res = re.findall('#\d+',line)
            # print(res)
            if res:
                # res.
                for pid2 in res:
                    tgt = int(pid2.replace('#',''))
                    if [pid,tgt] not in pos:        # detail中有描述但不在正例中
                        fout.write(str(pid)+' '+str(tgt)+'\n')

        f.close()
        fout.close()




    def get_l2l(self):
        from multiprocessing import Pool
        pool = Pool(10)
        f = open('../data/preprocess/places_hk_gps_new.txt','r',encoding='utf-8')
        self.places = {}
        for line in f:
            line = line.strip().split()
            if line[-1] =='0':
                continue
            self.places[int(line[2])] = [int(line[0]),float(line[3]),float(line[4])]
        f.close()
        
        
        l2ls = pool.map(self.g_l2l,list(self.places.keys()))

        # print(l2ls)
        f = open('../data/preprocess/l2l_hk_{}_{}.txt'.format(self.t_th,self.d_th),'w',encoding='utf-8')
        for l2l in l2ls:
            for key,values in l2l.items():
                for value in values:
                    f.write(str(key)+' '+str(value[0])+' '+str(value[1])+'\n')
        f.close()
        # l2ls = pool.map(self.g_l2l,places.keys())

        
        # print(self.l2l)
        
        # self.f.close()        
        # get location 2 loction file (location-id location-id weight)

    def make_csv(self):
        # make files for gephi visulizer
        f = open('idmap.csv','w',encoding='utf-8')
        for i in range(1017):
            f.write(str(i+1)+','+'p\n')
        for i in range(2388):
            f.write(str(i+1018)+','+'l\n')
        f.close()

        f = open('edges.csv','w',encoding='utf-8')
        f.write('source,target,weight\n')



        p2l = open('../data/p2l.txt','r',encoding='utf-8')
        for line in p2l:
            line = line.strip().split()
            f.write(str(line[0])+','+str(int(line[1])+1017)+',1\n')
        l2l = open('../data/l2l_hk_{}_{}.txt'.format(self.t_th,self.d_th),'r',encoding='utf-8')
        for line in l2l:
            line = line.strip().split()
            f.write(str(int(line[0])+1017)+','+str(int(line[1])+1017)+','+line[2]+'\n')
        f.close()
        l2l.close()
        p2l.close()

        return 0

    def address_filter(self,address):
        address = address.replace('（','(').replace(' ','')
        if '(' in address:
            if '座位' in address or '頭等艙' in address:
                address = address[:address.index('(')]
        return address

    def get_locations_3(self):
        # 删除skip中的地址
        # get location file  (location_id name time lng lat)
        f = open('../data/preprocess/places_hk.txt','r',encoding='utf-8')
        p2l = {}
        places = {}
        cnt = 1

        for idx,line in enumerate(f):
            if (idx+1) % 100 ==0:
                print('processed {} places'.format(idx+1))
            flag = False
            dates = []
            line = line.strip().split()


            if len(line) < 2:
                continue

            f = False
            for skip in self.skip:
                if skip in line:
                    # print(line)
                    f = True
            if f:
                continue            ## 跳过医院 航班等地址

            pid = line[0]
            if pid not in p2l:
                p2l[pid] = []
            address = self.address_filter(''.join(line[6:]))
            for date in line[1:5]:
                date = self.get_date(date.split('_')[-1])
                dates.append(date)
                                      ############此处后续可以修改
            if dates[2] != 0 and dates[3] != 0:
                start = dates[2]
                end = dates[3]

            elif dates[2] != 0 and dates[3] == 0:
                start = dates[2]
                end = dates[2]

            elif dates[3] !=0 and dates[2] ==0:
                start = dates[3]
                end = dates[3]
            
            elif dates[3] == 0 and dates[2] == 0:    
                if 'action_住宿' in line:
                    end = dates[1] if dates[1] != 0 else dates[0]
                    start = dates[0]-3 if dates[0] !=0 else end-3           
                else:
                    end = dates[0] if dates[0] != 0 else dates[1]
                    start = end - 3       #按发病前3天开始有传染性

            # elif 'action_住宿' in line:
            #     end = dates[1] if dates[1] != 0 else dates[0]
            #     start = dates[0]-3 if dates[0] !=0 else end-3

            # for skip in self.skip:
            #     if skip in line:
            #         flag = True
            #         end = start         
            #         for date in range(end,end+1):         #只有一天
            #             if str(date)+' '+address not in places: 
            #                 places[str(date)+' '+address] = [cnt,0,0]        
            #                 p2l[pid].append(cnt)
            #                 cnt += 1  
            #             else:
            #                 p2l[pid].append(places[str(date)+' '+address][0])
            #         break 
            # if flag:
            #     continue                                        
                      
            for _ in range(10):
                try:
                    lng,lat = self.google(address)
                    break
                except:
                    lng,lat = 0,0

            
            if lng == 0 and lat == 0:
                # end = start            
                #对于没有坐标的 跳过
                continue
                # print(address)
            for date in range(end,end+1):
                if str(date)+' '+address not in places:
                    places[str(date)+' '+address] = [cnt,lng,lat]  
                    p2l[pid].append(cnt)
                    cnt += 1 
                else:
                    p2l[pid].append(places[str(date)+' '+address][0])

        

        f = open('../data/preprocess/p2l_new.txt','w',encoding='utf-8')
        # print(p2l)
        for key,value in p2l.items():
            if len(value) > 0:
                for lid in value:
                    f.write(key + ' ' + str(lid)+'\n')
        f.close()

        f = open('../data/preprocess/places_hk_gps_new.txt','w',encoding='utf-8')
        for key,value in places.items():
            f.write(key + ' ')
            for item in value:
                f.write(str(item)+' ')
            f.write('\n')
        f.close()
        return 0 

    def get_locations_2(self):
        # 对于skip中的只保留一天
        # get location file  (location_id name time lng lat)
        f = open('../data/places_hk.txt','r',encoding='utf-8')
        p2l = {}
        places = {}
        cnt = 1

        for idx,line in enumerate(f):
            if (idx+1) % 100 ==0:
                print('processed {} places'.format(idx+1))
            flag = False
            dates = []
            line = line.strip().split()
            pid = line[0]
            if pid not in p2l:
                p2l[pid] = []
            address = self.address_filter(''.join(line[6:]))
            for date in line[1:5]:
                date = self.get_date(date.split('_')[-1])
                dates.append(date)
                                      ############此处后续可以修改
            if dates[2] != 0 and dates[3] != 0:
                start = dates[2]
                end = dates[3]

            if dates[2] != 0 and dates[3] == 0:
                start = dates[2]
                end = dates[2]

            if dates[3] !=0 and dates[2] ==0:
                start = dates[3]
                end = dates[3]
            
            if dates[3] == 0 and dates[2] == 0:               
                end = dates[0] if dates[0] != 0 else dates[1]
                start = end - 3       #按发病前3天开始有传染性
                
            for skip in self.skip:
                if skip in line:
                    flag = True
                    end = start         
                    for date in range(end,end+1):         #只有一天
                        if str(date)+' '+address not in places: 
                            places[str(date)+' '+address] = [cnt,0,0]        
                            p2l[pid].append(cnt)
                            cnt += 1  
                        else:
                            p2l[pid].append(places[str(date)+' '+address][0])
                    break 
            if flag:
                continue                                        
                      
            for _ in range(10):
                try:
                    lng,lat = self.google(address)
                    break
                except:
                    lng,lat = 0,0

            
            if lng == 0 and lat == 0:
                end = start             #对于没有坐标的仅保留一天 
                print(address)
            for date in range(end,end+1):
                if str(date)+' '+address not in places:
                    places[str(date)+' '+address] = [cnt,lng,lat]  
                    p2l[pid].append(cnt)
                    cnt += 1 
                else:
                    p2l[pid].append(places[str(date)+' '+address][0])

        

        f = open('../data/p2l.txt','w',encoding='utf-8')
        # print(p2l)
        for key,value in p2l.items():
            if len(value) > 0:
                for lid in value:
                    f.write(key + ' ' + str(lid)+'\n')
        f.close()

        f = open('../data/places_hk_gps.txt','w',encoding='utf-8')
        for key,value in places.items():
            f.write(key + ' ')
            for item in value:
                f.write(str(item)+' ')
            f.write('\n')
        f.close()
        return 0 

    def get_locations(self):
        # get location file  (location_id name time lng lat)
        f = open('../data/places_hk.txt','r',encoding='utf-8')
        p2l = {}
        places = {}
        cnt = 1

        for idx,line in enumerate(f):
            if (idx+1) % 100 ==0:
                print('processed {} places'.format(idx+1))
            flag = False
            dates = []
            line = line.strip().split()
            if len(line)<2:
                continue
            pid = line[0]
            if pid not in p2l:
                p2l[pid] = []
            address = self.address_filter(''.join(line[6:]))
            for date in line[1:5]:
                date = self.get_date(date.split('_')[-1])
                dates.append(date)
                                      ############此处后续可以修改
            if dates[2] != 0 and dates[3] != 0:
                start = dates[2]
                end = dates[3]

            if dates[2] != 0 and dates[3] == 0:
                start = dates[2]
                end = dates[2]

            if dates[3] !=0 and dates[2] ==0:
                start = dates[3]
                end = dates[3]
            
            if dates[3] ==0 and dates[2] == 0:               
                end = dates[0] if dates[0] != 0 else dates[1]
                start = end - 3       #按发病前3天开始有传染性


            if 'action_住宿' in line:
                end = dates[1] if dates[1] != 0 else dates[0]
                start = dates[0]-3 if dates[0] !=0 else end-3
                
            for skip in self.skip:
                if skip in line:
                    flag = True
                    end = start         
                    for date in range(start,end+1):
                        if str(date)+' '+address not in places: 
                            places[str(date)+' '+address] = [cnt,0,0]        
                            p2l[pid].append(cnt)
                            cnt += 1  
                        else:
                            p2l[pid].append(places[str(date)+' '+address][0])
                    break 
            if flag:
                continue                                        
                      
            for _ in range(10):
                try:
                    lng,lat = self.google(address)
                    break
                except:
                    lng,lat = 0,0

            
            if lng == 0 and lat == 0:
                end = start             #对于没有坐标的仅保留一天 
                print(address)
            for date in range(start,end+1):
                if str(date)+' '+address not in places:
                    places[str(date)+' '+address] = [cnt,lng,lat]  
                    p2l[pid].append(cnt)
                    cnt += 1 
                else:
                     p2l[pid].append(places[str(date)+' '+address][0])

        

        f = open('../data/p2l.txt','w',encoding='utf-8')
        # print(p2l)
        for key,value in p2l.items():
            if len(value) > 0:
                for lid in value:
                    f.write(key + ' ' + str(lid)+'\n')
        f.close()

        f = open('../data/places_hk_gps.txt','w',encoding='utf-8')
        for key,value in places.items():
            f.write(key + ' ')
            for item in value:
                f.write(str(item)+' ')
            f.write('\n')
        f.close()
        return 0 

    def new_attr(self):
        f = open('../data/attrs_hk_raw.txt','r',encoding='utf-8')
        attrs = []
        for line in f:
            attrs.append(line.strip())
        f.close()

        f = open('../data/places_hk.txt','r',encoding='utf-8')
        p2l = {}
        for line in f:
            line = line.strip().split()
            address = self.address_filter(''.join(line[6:]))
            if line[0] not in p2l:
                p2l[line[0]] = []
            p2l[line[0]].append(address)
            
        f.close()

        f = open('../data/attrs_hk.txt','w',encoding='utf-8')
        for key,value in p2l.items():
            if len(value) > 0:
                for lid in value:
                    tmp = ' location_'+ str(lid)
                    attrs[int(key)-1] += tmp
        for attr in attrs:
            f.write(attr+'\n')
        f.close()

    def get_p2p(self):
        p2p = []
        f = open('../data/preprocess/attrs_hk_raw.txt','r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            pid = line[0].split('_')[-1]
            relations = line[-1].split('_')[-1]
            if relations == 'None':
                continue
            else:
                relations = relations.split(',')
                for relation in relations:
                    if pid ==  relation:
                        continue
                    if [pid,relation] not in p2p:
                        p2p.append([pid,relation])
                    if [relation,pid] not in p2p:
                        p2p.append([relation,pid])

        f.close()
        f = open('../data/preprocess/doulink_hk.txt','w',encoding='utf-8')
        for pair in p2p:
            f.write(' '.join(pair)+'\n')
        f.close()
        # get patient 2 patient infection relations
        return 0

    def data_info(self):
        group = []
        cnt = 0
        f = open('../data/attrs_hk_raw.txt','r',encoding='utf-8')
        for line in f:
            line = line.strip().split()
            for attr in line:
                if 'group_id' in attr:
                    attr = attr.split('_')[-1]
                    if attr != 'None':
                        cnt += 1 
                        if attr not in group:
                            group.append(attr)
        print(group)
        print(cnt)
        f.close()


if __name__ == "__main__":
    # covid = CovidPreprocess(3,3)
    # # covid.make_csv()
    # print(covid.google('英國倫敦'))
    # covid.get_locations_3()
    # # covid.get_p2p()
    # # covid.data_info()
    # covid.get_l2l()
    # covid = CovidPreprocess(10,10)
    # covid.get_l2l()
    # covid.new_attr()


    covid = CovidPreprocess(3,3)
    # covid.get_locations_3()
    # covid.get_p2p()
    # covid.get_l2l()
    # covid = CovidPreprocess(10,10)
    # covid.get_l2l()
    covid.get_validset()