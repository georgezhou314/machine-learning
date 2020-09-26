import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

def split(path):
	im = cv2.imread(path)
	# 裁剪
	#im = im[3:27,0:70]
	im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	# 二值化
	ret,im_inv = cv2.threshold(im_gray,127,255,cv2.THRESH_BINARY_INV)
	# 降噪
	kernel = 1/16*np.array([[1,2,1], [2,4,2], [1,2,1]])
	im_blur = cv2.filter2D(im_inv,-1,kernel)
	# 再来一次二值化
	ret,im_res = cv2.threshold(im_blur,127,255,0)
	# cv2.imshow("img",im_res)
	# 轮廓的点集，轮廓的索引
	contours, hierarchy = cv2.findContours(im_res,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	w,h = 0,0
	split_cnt = 0
	size = []
	for i in range(len(contours)):
		x,y,w,h= cv2.boundingRect(contours[i])
		print(x,y,w,h)
		if w>2 and h>2:
			x = max(x-2,0)
			y = max(y-2,0)
			w = w+2
			h = h+2
			size.append([x,y,w,h])
			'''
			im_copy = im.copy()
			img  = cv2.rectangle(im_copy,(x,y),(x+w,y+h),(0,0,255),1)
			fileName = "img_"+str(int(time.time()*1000*1000))+".png"
			cv2.imwrite(fileName,img)
			print("save+1")
			'''
			split_cnt =split_cnt+1
	# 分成了三个结果，其中有一张图，包含两个字符
	result = []
	if split_cnt == 4:
		for contour in size:
			x,y,w,h = contour
			box = np.int0([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])
			result.append(box)
		print("分成四块")
	elif split_cnt==3:
		# 最大子图
		w_max = 0
		for tmp in size:
			if tmp[2] > w_max:
				w_max = tmp[2]
		for contour in size:
			x, y, w, h = contour
			if w == w_max: # w_max是所有contonur的宽度中最宽的值
				box_left = np.int0([[x,y], [x+w/2,y], [x+w/2,y+h], [x,y+h]])
				box_right = np.int0([[x+w/2,y], [x+w,y], [x+w,y+h], [x+w/2,y+h]])
				result.append(box_left)
				result.append(box_right)
			else:
				box = np.int0([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])
				result.append(box)
		print("box.size",len(result))
		print("分出三块")
	# 只分出了一个结果
	elif split_cnt==1:
		contour = size[0]
		x,y,w,h = contour
		box0 = np.int0([[x,y], [x+w/4,y], [x+w/4,y+h], [x,y+h]])
		box1 = np.int0([[x+w/4,y], [x+w*2/4,y], [x+w*2/4,y+h], [x+w/4,y+h]])
		box2 = np.int0([[x+w*2/4,y], [x+w*3/4,y], [x+w*3/4,y+h], [x+w*2/4,y+h]])
		box3 = np.int0([[x+w*3/4,y], [x+w,y], [x+w,y+h], [x+w*3/4,y+h]])
		result.extend([box0, box1, box2, box3])
		print("只分出一块")
		'''
		for box in result:
			im_copy = im.copy()
			img = cv2.rectangle(im_copy,tuple(box[0]),tuple(box[2]),(0,0,255),1)
			fileName = "img_"+str(int(time.time()*1000*1000))+".png"
			cv2.imwrite(fileName,img)
			print("save+1")
		'''
	# 仅有两种结果的情况
	elif split_cnt == 2:
		w_max,w_min = 0,1000
		for contour in size:
			if contour[3] > w_max:
				w_max = contour[3]
			if contour[3] < w_min:
				w_min = contour[3]
		for contour in size:
			print(contour)
			x,y,w,h = contour
			if w == w_max and w_max >= w_min * 2:
				# 如果两个轮廓一个是另一个的宽度的2倍以上，我们认为这个轮廓就是包含3个字符的轮廓
				box_left = np.int0([[x,y], [x+w/3,y], [x+w/3,y+h], [x,y+h]])
				box_mid = np.int0([[x+w/3,y], [x+w*2/3,y], [x+w*2/3,y+h], [x+w/3,y+h]])
				box_right = np.int0([[x+w*2/3,y], [x+w,y], [x+w,y+h], [x+w*2/3,y+h]])
				result.append(box_left)
				result.append(box_mid)
				result.append(box_right)
			elif w_max < w_min * 2:
				# 如果两个轮廓，较宽的宽度小于较窄的2倍，我们认为这是两个包含2个字符的轮廓
				box_left = np.int0([[x,y], [x+w/2,y], [x+w/2,y+h], [x,y+h]])
				box_right = np.int0([[x+w/2,y], [x+w,y], [x+w,y+h], [x+w/2,y+h]])
				result.append(box_left)
				result.append(box_right)
			else:
				box = np.int0([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])
				result.append(box)
			print("只分出两块,不保存")
		exit(-1)
	# 保存分块图
	for box in result:
		cv2.drawContours(im,[box],0,(0,0,255),1)
		roi = im_res[box[0][1]:box[3][1],box[0][0]:box[1][0]]
		roistd = cv2.resize(roi,(18,18))
		fileName = "number/split/"+str(int(time.time()*1000*1000))+".png"
		cv2.imwrite(fileName,roistd)
args = list(sys.argv)
i = 1
while i < len(args):
	print(args[i])
	split(args[i])
	i = i+1
