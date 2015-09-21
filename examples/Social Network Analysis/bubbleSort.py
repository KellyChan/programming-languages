def bubbleSort(list):
    for index in range(len(list)-1,0,-1):
        for i in range(index):
            if list[i]>list[i+1]:
                temp = list[i]
                list[i] = list[i+1]
                list[i+1] = temp
                
unorderList = [5, 19, 1, 3, 45, 98, 77, 100, 2]
bubbleSort(unorderList)
print(unorderList)