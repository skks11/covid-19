
class preprocesser:
    def __init__(self,l2l_file,p2l_file,outfile):
        self.mapping = {}
        self.l2l = l2l_file
        self.p2l = p2l_file
        self.outfile = outfile

    def get_mappings(self,location_num,patient_num):
        f = open('node_mappings.txt','w',encoding = 'utf-8')
        for i in range(patient_num):
            f.write(str(i+1),' patient\n')
        for j in range(location_num):
            self.mapping[str(j+1)] = str(j+patient_num+1)
            f.write(str(j+patient_num+1),' lcoation\n')
        f.close()

    # 注意是否需要双向
    def get_non_HIN(self,l2l,p2l):
        fout = open(self.outfile,'w',encoding='utf-8')
        
        f = open(self.l2l,'r',encoding='utf-8',encoding='utf-8')
        for line in f:
            line = line.strip.split()
            self.outfile.write(self.mapping(line[0])+' '+self.mapping(line[1])+' '+line[2]+'\n')
        f.close()

        f = open(self.l2l,'r',encoding='utf-8',encoding='utf-8')
        for line in f:
            line = line.strip.split()
            self.outfile.write(line[0]+' '+self.mapping(line[1])+' '+line[2]+'\n')
            self.outfile.write(self.mapping(line[1])+' '+line[0]+' '+line[2]+'\n')
        f.close()

        fout.close()

         


