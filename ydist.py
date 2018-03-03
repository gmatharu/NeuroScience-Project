import bz2,pickle,os
os.chdir('/storage/hpc_gagand87/processed_data/')
areafiles = list(filter(None, os.popen('find avg_per_sub -name *.pkl.bz2').read().split('\n')))
print(areafiles)
area_count = []
for f in areafiles:
	with bz2.BZ2File(f, 'r') as handle:
        	baai = pickle.load(handle)
	for i in baai.avg_int:
		area_count.append([area for area in i[0].keys() if i[0][area] == 1][0]) 
outfile= open('area_counts.txt','w')
for item in area_count:
	outfile.write("%s\n" % item)
outfile.close()
	
