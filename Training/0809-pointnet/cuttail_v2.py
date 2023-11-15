import os
data=[]
dn=input("input data dir:")
tn=input("output test data dir:")
on=input("output data dir name:")

if tn not in os.listdir("."):
    os.mkdir(tn)

if on not in os.listdir(tn):
    os.mkdir(tn+"\\"+on+"_label")
    os.mkdir(tn+"\\"+on+"_points")


for file in os.listdir(dn):
    print(dn+"/"+file)
    data.append(file)
for name in data:
    f=open(dn+"/"+name,'r')
    points=f.read().split("\n")
    f.close()
    print(points)
    for i in range(0,len(points)-1):
        cut=points[i].split(" ")
        print(int(float(cut[3])))
        typename=name.split(".")
        if i == 0:
            f=open(tn+"\\"+on+"_points/"+name,'w')
        else:
            f=open(tn+"\\"+on+"_points/"+name,'a')
        f.write(cut[0]+" "+cut[1]+" "+cut[2]+"\n")
        f.close()
        if i == 0:
            fd=open(tn+"\\"+on+"_label/"+typename[0]+".seg",'w')
        else:
            fd=open(tn+"\\"+on+"_label/"+typename[0]+".seg",'a')
        fd.write(str(int(float(cut[3])))+"\n")
        fd.close()