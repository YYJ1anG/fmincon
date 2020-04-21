from sympy import *
import numpy as np
from fmincon import *

w0 = np.array([1,1])#初始点**********************别忘了写这个！！！！
n = np.size(w0)
for i in range(n):#动态声明变量
	exec('str1 = \'x{}\'.format(i)')
	exec('x{} = Symbol(str1)'.format(i))


'''
fmincon:

objective function:
	min z
s.t.:
	g(x)>=0
	h(x)=0
'''
z = x0**2 + 2*x1**2 #目标函数
g = [x0+x1-1]
h = []


w1,z = fmincon(z,w0,g,h)
for i in range(n):
	print("x{}: ".format(i) + str(w1[i].evalf()))

replace = []
for i in range(n):#计算replace
	replace.append( ['x{}'.format(i),w1[int('{}'.format(i))]])  
print('z： ' + str(z.subs(replace).evalf()))