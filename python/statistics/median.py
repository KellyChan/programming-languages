def median(lst):
    lst.sort()
    if len(lst) % 2 != 0: #Odd number of terms
        return lst[int((len(lst)-1)/2)]
    else: #Even number of terms
        term1 = lst[int((len(lst)/2)-1)]
        term2 = lst[int((len(lst)/2))]
        return (term1 + term2)/2.0
