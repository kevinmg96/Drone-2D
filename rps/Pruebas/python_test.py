import numpy as np

x = [1,2]
y = [1,2,3]
print(x == y)

x = np.random.randint(0,9,size=(3,4))
print(x)
y = np.random.randint(0,9,size=(3,4))
print(y)
w = np.concatenate([x,y],axis = 1)
print(w)
boundaries = [0,0,10,10]
bool_arr = np.zeros([w.shape[1]])
s = w[0,:] > 12

c_1 = (w[0,:] > boundaries[2]).any()
c_2 = (w[0,:] < boundaries[0]).any()
c_3 = (w[1,:] > boundaries[3]).any()
c_4 = (w[1,:] < boundaries[1]).any()
r = np.where(c_1 or c_2 or c_3 or c_4,True,False)

print(r)

bool_new_direction = np.ones([5,]).astype(bool) 
print(bool_new_direction)

class parent:
    def __init__(self,**kwargs) -> None:
        print("hey")
        self.rand_num = kwargs["RandNum"]
        self.bool_test = kwargs["Bool"]
    def print_data(self):
        print(self.rand_num,self.bool_test)
class child(parent):
    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)

c = child(RandNum = 10,Bool = True)
c.print_data()
print("yoo")
print(~-False)
        