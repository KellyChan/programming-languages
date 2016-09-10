-- Project: Dell DVD Store Database Test Suite
-- Author: Kelly Chan
-- Date: May 6 2014

-- Data Link 1: http://linux.dell.com/dvdstore/
-- Data Link 2: https://s3-us-west-2.amazonaws.com/selection-tasks/ds2-small.zip

-- NOTE: Queries below are based on Data Link 2
-- Q2. Which category of movies has sold the most DVDs?
-- Answer: Foreign (ID: 9)

-- solution 1. return CATEGORY, totalSALES

select CATEGORY, sum(SALES) as totalSALES 
from (
       select ds2.inventory.PROD_ID, ds2.inventory.SALES, 
               ds2.products.CATEGORY
       from ds2.inventory
       left join ds2.products
       on ds2.inventory.PROD_ID = ds2.products.PROD_ID) as temp

group by CATEGORY
order by totalSALES desc;


-- solution 2. return CATEGORYNAME, CATEGORY, totalSALES

select ds2.categories.CATEGORYNAME, 
        sales.CATEGORY, sales.totalSALES
from (
       select CATEGORY, sum(SALES) as totalSALES 
       from (

              select ds2.inventory.PROD_ID, ds2.inventory.SALES, 
			          ds2.products.CATEGORY
              from ds2.inventory
              left join ds2.products
			  on ds2.inventory.PROD_ID = ds2.products.PROD_ID) as temp

       group by CATEGORY
       order by totalSALES desc) as sales

left join ds2.categories
on sales.CATEGORY = ds2.categories.CATEGORY;


-- solution 3. return CATEGORYNAME, MAXSALES

select CATEGORYNAME, max(totalSALES) as MAXSALES 
from (

       select ds2.categories.CATEGORYNAME, sales.CATEGORY, sales.totalSALES
       from (
              select CATEGORY, sum(SALES) as totalSALES 
              from (

                     select ds2.inventory.PROD_ID, ds2.inventory.SALES, 
                             ds2.products.CATEGORY
                     from ds2.inventory
                     left join ds2.products
                     on ds2.inventory.PROD_ID = ds2.products.PROD_ID) as temp

		group by CATEGORY
		order by totalSALES desc) as sales

left join ds2.categories
on sales.CATEGORY = ds2.categories.CATEGORY) as maxsales;