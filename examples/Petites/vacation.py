def hotel_cost(nights):
    return nights * 140

def plane_ride_cost(city):
    if (city == "Charlotte"):
        return 183
    elif (city == "Tampa"):
        return 220
    elif (city == "Pittsburgh"):
        return 222
    elif (city == "Los Angeles"):
        return 475

def rental_car_cost(days):
    cost = days * 40
    
    if (days >= 3 and days < 7):
        return cost - 20
    elif (days >= 7):
        return cost - 50
    elif (days == 0):
        return 0
    else: 
        return cost
        
def trip_cost(city, days, spending_money):
    return spending_money + hotel_cost(days) + plane_ride_cost(city) + rental_car_cost(days)
    
print trip_cost("Los Angeles", 5, 600)
