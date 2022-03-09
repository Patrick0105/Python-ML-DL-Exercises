########基本運算複習########

#Introducing every kind of sets
x = [1, 2, 3] #list
y = (1, 2, 3) #tuple
z = {'key1': 1, 'key2': 2, 'key3': 3}   #dict
print(type(x), type(y), type(z))

#Introducing functions
print(x, type(x))
print('max =', max(x))
print('min =', min(x))
print(abs(-1.1), abs(1.1))
import math
print(math.sqrt(4))

#Introducing operators
print(1+3) #加法
print(1-3) #減法
print(1*3) #乘法
print(1/3) #除法
print(1//3) #取商
print(1%3) #取餘數

#Introducing type of variables
x = 1
y = 1.1
z = 'oh'
a = False
type(x), type(y), type(z), type(a)

#Introducing "+"
print(z+'2')
print('hello'+'world')
print(a+2)