def selectionSort1(list):
    for index in range(len(list)-1,0,-1):
        indexMax = 0
        for i in range(1,index+1):
            if list[i] > list[indexMax]:
                indexMax = i
                
        temp           = list[index]
        list[index]    = list[indexMax]
        list[indexMax] = temp
 
 
def selectionSort2(list):
    for index in range(0,len(list)-1):
        indexMin = 0
        for i in range(index+1,len(list)):
            if list[i] < list[indexMin]: indexMin = i
        if indexMin > index:
            temp           = list[index]
            list[index]    = list[indexMin]
            list[indexMin] = temp
        
    
    
unorderList = [34, 2, 32, 1, 0, 3, 6, 89, 66]
 
selectionSort1(unorderList)
print(unorderList)
 
selectionSort2(unorderList)
print(unorderList)