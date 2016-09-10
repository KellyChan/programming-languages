-- Project: Dell DVD Store Database Test Suite
-- Author: Kelly Chan
-- Date: May 6 2014

-- Data Link 1: http://linux.dell.com/dvdstore/
-- Data Link 2: https://s3-us-west-2.amazonaws.com/selection-tasks/ds2-small.zip

-- NOTE: Queries below are based on Data Link 2
-- Q1. How many customers have placed more than 1 order?
-- Answer: 2447


select count(CUSTOMERID) as total 
from (
       select CUSTOMERID, count(ORDERID) as NUM 
       from ds2.orders
       group by CUSTOMERID
	   having NUM > 1) as temp;