/*

PROGRAM NAME: Learning SAS by Example
Chapter 14: Displaying Your Data

PROGRAMMER: Kelly Chan
Date Written: Aug 16 2013

*/

libname learn 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\libs';

data learn.sales;
      infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\Sales.csv' dsd;
	  input EmpID $	Name $	Region $ Customer $	Item $	Quantity	UnitCost	TotalSales;
run;

/* PROC PRINT using all the defaults
*/
title "Listing of SALES";
proc print data=learn.sales;
run;

/* Controlling which variables appear in the listing
*/
title "Listing of SALES";
proc print data=learn.sales;
     var EmpID Customer TotalSales;
run;

/* Using an ID statement to omit the Obs column
*/
title "Listing of SALES";
proc print data=learn.sales;
     id EmpID;
     var Customer TotalSales;
run;

/* Adding a FORMAT statement to PROC PRINT
*/

proc print data=learn.sales;
      title "Listing of SALES";
      id EmpID;
      var Customer Quantity TotalSales;
      format TotalSales dollar10.2 
             Quantity comma7.;
run;

/* Controlling which observations appear in the listing (WHERE statement)
*/

title "Listing of SALES with Quantities greater than 400";
proc print data=learn.sales;
      where Quantity gt 400;
      id EmpID;
      var Customer Quantity TotalSales;
      format TotalSales dollar10.2 Quantity comma7.;
run;


/* Using the IN operator in a WHERE statement
*/

title "Listing of SALES with Quantities greater than 400";
proc print data=learn.sales;
       where EmpID in ('1843' '0177');
       id EmpID;
       var Customer Quantity TotalSales;
       format TotalSales dollar10.2 Quantity comma7.;
run;

/* Adding titles and footnotes to your listing
*/

title1 "The XYZ Company";
title3 "Sales Figures for Fiscal 2006";
title4 "Prepared by Roger Rabbit";
title5 "-----------------------------";
footnote "All sales figures are confidential";
proc print data=learn.sales;
      id EmpID;
      var Customer Quantity TotalSales;
      format TotalSales dollar10.2 Quantity comma7.;
run;

/* Using PROC SORT to change the order of your observations
*/

proc sort data=learn.sales;
      by TotalSales;
run;

/* Demonstrating the DESCENDING option of PROC SORT
*/


proc sort data=learn.sales out=sales;
      by descending TotalSales;
run;
title "Listing of SALES";
proc print data=learn.sales;
      id EmpID;
      var Customer Quantity TotalSales;
      format TotalSales dollar10.2 Quantity comma7.;
run;

/* Sorting by more than one variable
*/

proc sort data=learn.sales out=sales;
      by EmpID descending TotalSales;
run;
title "Sorting by More than One Variable";
proc print data=sales;
      id EmpID;
      var TotalSales Quantity;
      format TotalSales dollar10.2 Quantity comma7.;
run;

/* Using labels as column headings with PROC PRINT
*/

title "Using Labels as Column Headings";
proc print data=sales label;
     id EmpID;
     var TotalSales Quantity;
     label EmpID = "Employee ID"
           TotalSales = "Total Sales"
           Quantity = "Number Sold";
     format TotalSales dollar10.2 Quantity comma7.;
run;


/* Using a BY statement in PROC PRINT
*/

proc sort data=learn.sales out=sales;
      by Region;
run;

title "Using Labels as Column Headings";
proc print data=sales label;
      by Region;
      id EmpID;
      var TotalSales Quantity;
      label EmpID = "Employee ID"
            TotalSales = "Total Sales"
            Quantity = "Number Sold";
      format TotalSales dollar10.2 Quantity comma7.;
run;

/* Adding totals and subtotals to your listing
*/

proc sort data=learn.sales out=sales;
     by Region;
run;

title "Adding Totals and Subtotals to Your Listing";
proc print data=sales label;
     by Region;
     id EmpID;
     var TotalSales Quantity;
     sum Quantity TotalSales;
     label EmpID = "Employee ID"
           TotalSales = "Total Sales"
           Quantity = "Number Sold";
     format TotalSales dollar10.2 Quantity comma7.;
run;

/* Adding totals and subtotals to your listing
*/
proc sort data=learn.sales out=sales;
     by EmpID;
run;
title "Using the Same Variable in an ID and BY Statement";
proc print data=sales label;
     by EmpID;
     id EmpID;
     var Customer TotalSales Quantity;
     label EmpID = "Employee ID"
           TotalSales = "Total Sales"
           Quantity = "Number Sold";
     format TotalSales dollar10.2 Quantity comma7.;
run;

/* Demonstrating the N= option with PROC PRINT
*/

title "Demonstrating the N option of PROC PRINT";
proc print data=sales n="Total number of Observations:";
     id EmpID;
     var TotalSales Quantity;
     label EmpID = "Employee ID"
           TotalSales = "Total Sales"
           Quantity = "Number Sold";
     format TotalSales dollar10.2 Quantity comma7.;
run;

/* Double-spacing your listing
*/

title "Double-Spacing Your Listing";
proc print data=sales double;
     id EmpID;
     var TotalSales Quantity;
     label EmpID = "Employee ID"
           TotalSales = "Total Sales"
           Quantity = "Number Sold";
     format TotalSales dollar10.2 Quantity comma7.;
run;

/* Listing the first five observations of your data set
*/

title "First Five Observations from SALES";
proc print data=learn.sales(obs=5);
run;

/* Forcing variable labels to print horizontally
*/

title "First Five Observations from SALES";
proc print data=learn.sales(obs=5) heading=horizontal;
run;

