from sympy import *
import numpy as np



def fmincon(z,w0,g,h):
	x = Symbol('x')
	f1 = x**2
	f2 = 0
	f3 = Piecewise((f2,x<0),(f1,x>=0))
	s1 = f1

	p = 0
	for i in range(np.size(g)):
		p = p + f3.subs(x,-1*g[i])
	for i in range(np.size(h)):
		p = p + s1.subs(x,h[i])

	#开始
	lamda = 2
	epsilon = 1e-5
	c = 10
	k = 0
	n = np.size(w0)
	while True:
		z = z + lamda*p

		w1,z = gradientdes(z, w0)


		replace = []
		for i in range(n):#计算replace
			replace.append( ['x{}'.format(i),w1[int('{}'.format(i))]])
		k = k+1
		if p.subs(replace).evalf() < epsilon:#达到精度，w0为最小值点
			print(k,"轮梯度下降后找到最优解")
			return w1,z

		lamda = lamda*c
		

def gradientdes(z,w0):

	epsilon = 1e-6#精度
	k = Symbol('k')#步长

	n = np.size(w0)



	num = 0#迭代次数

	while (True):
		replace = []
		p = [1]#初始化，无意义
		for i in range(n):#计算replace
			replace.append( ['x{}'.format(i),w0[int('{}'.format(i))]])
		for i in range(n):#计算梯度

			p = np.column_stack((p,diff(z,'x{}'.format(i)).subs(replace)))

		p = -1*p[0,1:]#负梯度方向


		if((np.sum(np.square(p)))**0.5<epsilon):#负梯度的模长达到精度要求时退出
			break
	    
	    #用求导的方式，得到最优步长a

		w1 = p*k + w0
		
		replace = []
		for i in range(n):#计算replace
			replace.append( ['x{}'.format(i),w1[int('{}'.format(i))]])
		f = z.subs(replace)

		a=solve(diff(f,k),k)#对f(k)求导,并使其等于0，求出k


		w1=p*a[0]+w0
	    
		#print(w1[0].evalf(),w1[1].evalf(),z.subs(replace).evalf() )#观察迭代
	    
		if((np.sum(np.square(w1-w0)))**0.5<epsilon):#经过迭代后无明显变化时退出
			break
	    
		w0 =  w1
		num = num +1



	replace = []
	for i in range(n):#计算replace
		replace.append( ['x{}'.format(i),w1[int('{}'.format(i))]])  
	print ("本轮梯度下降",num,"步")
	#return round(w1[0].evalf()),round(w1[1].evalf()),round(z.subs(replace).evalf()) 
	return w1,z