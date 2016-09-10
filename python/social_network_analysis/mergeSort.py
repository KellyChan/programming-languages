def mergeSort(list):
    print("Splitting ", list)
    if len(list)>1:
        mid = len(list)//2
        leftHalf = list[:mid]
        rightHalf = list[mid:]
        
        mergeSort(leftHalf)
        mergeSort(rightHalf)
        
        index = 0
        indexLeft = 0
        indexRight = 0
        
        while indexLeft < len(leftHalf) and indexRight < len(rightHalf):
            if leftHalf[indexLeft] < rightHalf[indexRight]:
                list[index] = leftHalf[indexLeft]
                indexLeft = indexLeft + 1
            else:
                list[index] = rightHalf[indexRight]
                indexRight = indexRight + 1
            index = index + 1
        
        while indexLeft < len(leftHalf):
            list[index] = leftHalf[indexLeft]
            indexLeft = indexLeft + 1
            index = index + 1
        
        while indexRight < len(rightHalf):
            list[index] = rightHalf[indexRight]
            indexRight = indexRight + 1
            index = index + 1
 
    print("Merging ", list)
 
 
unorderList = [1, 34, 22, 10, 9, 56]
mergeSort(unorderList)
print(unorderList)