'''
使用蒙特卡洛模拟计算PI
'''

import random
circle_cnt = 0
cnt = 10000000
for i in range(cnt):
	x = random.uniform(-1,1)
	y = random.uniform(-1,1)
	if x**2+y**2 <= 1:
		circle_cnt += 1
print("PI",4*circle_cnt/cnt)
