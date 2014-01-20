def digit_sum(n):
    n = abs(n)
    sum_n = 0
    for i in range(len(str(n))):
        digit = n % 10
        n = n / 10
        sum_n += digit
    return sum_n
