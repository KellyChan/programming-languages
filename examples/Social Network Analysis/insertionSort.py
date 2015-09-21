def insertionSort1(list):
    for index in range(1, len(list)):
        value = list[index]
        i = index - 1
        
        while i >= 0:
            if value < list[i]:
                list[i+1] = list[i]
                list[i] = value
                i = i - 1
            else:
                break
 
 
 
def insertionSort2(list):
    for index in range(0,len(list)-1):
        value = list[index]
        i = index
        while(i > 0 and list[i-1] > value):
            list[i] = list[i-1]
            i = i - 1
        list[i] = value
       
       
 
unorderList = [12, 45, 1, 23, 9, 7]
 
insertionSort1(unorderList)
print(unorderList)
 
insertionSort2(unorderList)
print(unorderList)
