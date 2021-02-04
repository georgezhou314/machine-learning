import requests
import time
def request_one():
	url = "http://fdinfo.scu.edu.cn/WFManager/loginAction_getCheckCodeImg.action"
	r = requests.get(url)
	name = str(int(time.time()*1e6))
	fileName = "/home/george/number/"+name+".jpg"
	with open(fileName,"wb") as f:
		f.write(r.content)
		f.close()
	print("status:",r.status_code,"保存成功")
for i in range(50):
	request_one()
