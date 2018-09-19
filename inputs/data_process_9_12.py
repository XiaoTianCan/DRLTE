#coding=utf-8

import sys

filename = sys.argv[1]
ifile = open(filename, "r")
ofile = open(filename[:-4] + 'b' + ".txt", "w" )
contents = ifile.readlines()
print contents[-1]
fline = map(float, contents[-1].strip().split(','))
maxnum = max(fline)
for i in range(len(fline)):
    #fline[i] = round(fline[i]/2.2, 2) #added by gn 2018.9.14
    #fline[i] = round(fline[i] * 1.6, 2)
    fline[i] = round(fline[i] / maxnum * 2, 2)  
fline = ','.join(map(str,fline))
print fline
ofile.writelines(contents[:-1])
ofile.write(fline)
