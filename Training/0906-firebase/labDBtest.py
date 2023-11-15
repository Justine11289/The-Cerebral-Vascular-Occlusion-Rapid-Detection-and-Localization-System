from firebase import firebase

url = 'https://cgu-db-547f1-default-rtdb.firebaseio.com/'

fb = firebase.FirebaseApplication(url, None)

students= [{'stuid':'B000001','name':'大雄','Math':'90','English':'88','Art':'89'},{'stuid':'B000002','name':'靜香','Math':'86','English':'58','Art':'73'},{'stuid':'B000003','name':'胖虎','Math':'74','English':'75','Art':'84'}]

for student in students:
    fb.post('students',student)
    print("{} 儲存完畢".format(student))


list = fb.get("/students/",None)

for key,value in list.items():
    print("key={}\tstuid={}\tname={}\tMath{}\tEnglish{}\tArt{}".format(key,value["stuid"],value["name"],value["Math"],value["English"],value["Art"]))


for key,value in list.items():
    if value["name"] == "大雄":
        fb.delete('/students/',key)