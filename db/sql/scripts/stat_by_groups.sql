select 
Timesheet.Item as ID, 
DictTimesheet.Item as Item, 
count(Timesheet.Date) as N, 
sum(Timesheet.ItemHours) as TotalHours, 
avg(Timesheet.ItemHours) as MeanHours 

from Timesheet, DictTimesheet
where Timesheet.Item = DictTimesheet.IDItem
group by ID