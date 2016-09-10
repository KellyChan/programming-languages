select 
Type,
count(Date) as Days, 
sum(Hours) as TotalHours, 
avg(Hours) as MeanHours 

from lifeTracker 
group by Type
order by count(*) desc
