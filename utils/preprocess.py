
class preprocesser:
    def __init__(self,l2l_file,p2l_file,outfile):
        self.mapping = {}
        self.l2l = l2l_file
        self.p2l = p2l_file
        self.outfile = outfile
        self.location_num = 2388
        self.patient_num = 1017

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




if __name__ == '__main__':
    P = preprocesser('../data/l2l.txt','../data/p2l.txt','../data/edge.txt')
    P.get_edge()

    P = preprocesser('../data/l2l.txt','../data/p2l.txt','../data/edge_nonHINE.txt')
    P.get_non_HIN()
         


