text_file = open('/home/hla115/sfuhome/cmpt318/project/img_txt/2016-06-05 06_00.txt','r')
wfile = open('wfile','w')
lines = text_file.read().split(',')
for i in range (0,len(lines),3):
	string = lines[i]+','+lines[i+1]+','+lines[i+2]+'\n'
	wfile.write(string)


