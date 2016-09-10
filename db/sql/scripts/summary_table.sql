select 
Year, 
Month,
count(Date) as NumberOfItems, 
sum(Price) as MonthlyTotal 

from Expense
group by Year, Month
order by Year desc, Month desc
