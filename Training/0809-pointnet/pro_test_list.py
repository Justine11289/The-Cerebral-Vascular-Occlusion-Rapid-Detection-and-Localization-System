import os

test_data_dir=raw_input("input test dir name:")
print test_data_dir
test_data_list=[]
dirname = os.listdir(test_data_dir)
print dirname
classify=dirname[0].split("_")
if classify[1] == "points":
	for file in os.listdir(test_data_dir+"/"+dirname[0]):
		print file
		onlyname=file.split(".")
		test_data_list.append(classify[0]+"_points/"+file+" "+classify[0]+"_label/"+onlyname[0]+".seg 02691156\n")
print test_data_list
f=open("testing_ply_file_list.txt","w")
f.writelines(test_data_list)
f.close()