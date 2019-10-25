import numpy as np

#quickstore算法
def quickstore(arr):#******
    if len(arr)<=1:
        return arr
    pivot=arr[len(arr)//2]#//取整 /会导致float代替整数
    left=[x for x in arr if x<pivot]
    middle=[x for x in arr if x==pivot]
    right=[x for x in arr if x>pivot]
    return quickstore(left)+middle+quickstore(right)

if __name__=="__main__":
    print(quickstore([3,6,8,10,1,2,1]))#[1, 1, 2, 3, 6, 8, 10]

    hello='hello'
    world='world'
    print(hello)
    print(len(hello))
    hw=hello+' '+world
    print(hw)
    hw12='%s %s %d'% (hello,world,12)
    print(hw12)

#list
    xs=[3,1,2]
    print(xs,xs[2])
    print(xs[-1])
    xs[2]='foo'
    print(xs)
    xs.append('bar')
    print(xs)
    x=xs.pop()
    print(x,xs)#bar [3, 1, 'foo']拿出并取出最后一个值

#slicing
    nums=list(range(5))#range返回range object所以想要返回列表加上list
    print(nums)
    print(nums[2:4])#不包括第四个
    print(nums[2:])
    print(nums[:2])
    print(nums[:])
    print(nums[:-1])#到最后一个但不包括最后一个
    nums[2:4]=[8,9]
    print(nums)

#loops
    animals=['cat','dog','monkey']
    for animal in animals:
        print(animal)
    #如果想要在循环体内访问每个元素的指针，可以使用内置的enumerate函数
    animals=['cat','dog','monkey']
    for idx,animal in enumerate(animals):
        print('#%d:%s' % (idx+1,animal))

#List comprehensions列表推导
    nums=list(range(5))
    squares=[]
    for x in nums:
        squares.append(x**2)
    print(squares)
    #使用列表推导可以使代码简化很多
    nums=list(range(5))
    squares=[x**2 for x in nums]
    print(squares)#[0, 1, 4, 9, 16]
    #条件
    nums=[0,1,2,3,4]
    even_squares=[x **2 for x in nums if x%2==0]
    print(even_squares)#[0, 4, 16]

#dictionaries
    d={'cat':'cute','dog':'furry'}
    print(d['cat'])
    print('cat' in d)#Check if a dictionary has a given key; prints "True"
    d['fish']='wet'
    print(d['fish'])
    print(d.get('monkey','N/A'))#Get an element with a default; prints "N/A"
    print(d.get('fish','N/A'))#Get an element with a default; prints "wet"
    del d['fish']
    print(d.get('fish','N/A'))

#loops
    d={'person':2,'cat':4,'spider':8}
    for animal in d:
        legs=d[animal]
        print('A %s has %d legs' % (animal,legs))
    #如果你想要访问键和对应的值，那就使用iteritems方法#Python 3.x 里面，
    # iteritems() 和 viewitems() 这两个方法都已经废除了，
    # 而 items() 得到的结果是和 2.x 里面 viewitems() 一致的。
    # 在3.x 里 用 items()替换iteritems() ，可以用于 for 来循环遍历。
    d = {'person': 2, 'cat': 4, 'spider': 8}
    for animal,legs in d.items():
        print('A %s has %d legs' % (animal, legs))
    nums=[0,1,2,3,4]
    even_num_to_square={x:x**2 for x in nums if x%2==0 }
    print(even_num_to_square)

#元组Tuples元组是一个值的有序列表（不可改变）。
    # 从很多方面来说，元组和列表都很相似。
    # 和列表最重要的不同在于，元组可以在字典中用作键，
    # 还可以作为集合的元素，而列表不行。
    d={(x,x+1):x for x in range(10)}
    print(d)
    t=(5,6)
    print(type(t))
    print(d[t])
    print(d[(1,2)])

#一个numpy数组是一个由不同数值组成的网格。
#网格中的数据都是同一种数据类型，
#可以通过非负整型数的元组来访问。
#维度的数量被称为数组的阶，
#数组的大小是一个由整型数构成的元组,
#可以描述数组不同维度上的大小。
    a=np.array([1,2,3])
    print(type(a))
    print(a.shape)#(3,)
    print(a[0],a[1],a[2])
    a[0]=5
    print(a)

    b=np.array([[1,2,3],[4,5,6]])
    print(b)#[[1 2 3]
             #[4 5 6]]
    print(b.shape)#(2, 3)
    print(b[0][0],b[0][1],b[1][0])

#访问数组：和Python列表类似，numpy数组可以使用切片语法。
#因为数组可以是多维的，所以你必须为每个维度指定好切片。
    a=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    b=a[:2,1:3]
    print(a[0][1])#2
    b[0][0]=77
    print(a[0][1])#7
    #数组的切片是对相同数据的视图，因此修改它将修改原始数组。

    #!!!你可以同时使用整型和切片语法来访问数组。
    #但是，这样做会产生一个比原数组低阶的新数组。
    a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    row_r1=a[1,:]
    row_r2=a[1:2,:]
    print(row_r1,row_r1.shape)#[5 6 7 8] (4,) rank为1
    print(row_r2,row_r2.shape)#[[5 6 7 8]] (1, 4) rank为2

#整型数组访问：当我们使用切片语法访问数组时，得到的总是原数组的一个子集。
#整型数组访问允许我们利用其它数组的数据构建一个新的数组
    a=np.array()



















