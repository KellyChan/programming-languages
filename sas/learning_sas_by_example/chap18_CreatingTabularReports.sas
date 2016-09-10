/*

PROGRAM NAME: Learning SAS by Example
Chapter 18: Creating Tabular Reports

PROGRAMMER: Kelly Chan
Date Written: Aug 16 2013

*/

libname learn 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\libs';
libname myfmts 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\formats';
options fmtsearch=(myfmts);

/* PROC TABULATE with all the defaults and a single CLASS variable
*/

title "All Defaults with One CLASS Variable";
proc tabulate data=learn.blood;
     class Gender;
     table Gender;
run;

/* Demonstrating concatenation with PROC TABULATE
*/

title "Demonstrating Concatenation";
proc tabulate data=learn.blood format=6.;
      class Gender Blood;
      table Gender Blood;
run;

/* Demonstrating table dimensions with PROC TABULATE
*/

title "Demonstrating Table Dimensions";
proc tabulate data=learn.blood format=6.;
      class Gender Blood;
      table Gender, Blood; /* cross tab*/
run;

/* Demonstrating the nesting operator with PROC TABULATE
*/

title "Demonstrating Nesting";
proc tabulate data=learn.blood format=6.;
     class Gender Blood;
     table Gender * Blood;
run;


/* Adding the keyword ALL to your table request
*/


title "Adding the Keyword ALL to the TABLE Request";
proc tabulate data=learn.blood format=6.;
    class Gender Blood;
    table Gender ALL,
          Blood  ALL;
run;

/* Using PROC TABULATE to produce descriptive statistics
*/

title "Demonstrating Analysis Variables";
proc tabulate data=learn.blood;
    var RBC WBC;
    table RBC WBC;
run;

/* Specifying statistics on an analysis variable with PROC TABULATE
*/

title "Specifying Statistics";
proc tabulate data=learn.blood;
     var RBC WBC;
     table RBC*mean WBC*mean;
run;

/* Specifying more than one descriptive statistic with PROC TABULATE
*/

title "Specifying More than One Statistic";
proc tabulate data=learn.blood format=comma9.2;
      var RBC WBC;
      table (RBC WBC)*(mean min max);
run;

/* Combining CLASS and analysis variables in a table
*/

title "Combining CLASS and Analysis Variables";
proc tabulate data=learn.blood format=comma11.2;
      class Gender AgeGroup;
      var RBC WBC Chol;
      table (Gender ALL)*(AgeGroup All),(RBC WBC Chol)*mean; /*cross tab*/
run;

/* Associating a different format with each variable in a table
*/

title "Specifying Formats";
proc tabulate data=learn.blood;
    var RBC WBC;
    table RBC*mean*f=7.2 
          WBC*mean*f=comma7.;
run;

/* Renaming keywords with PROC TABULATE
*/

title "Specifying Formats and Renaming Keywords";
proc tabulate data=learn.blood;
    class Gender;
    var RBC WBC;
    table Gender ALL,RBC*(mean*f=9.1    std*f=9.2)
                     WBC*(mean*f=comma9. std*f=comma9.1);
    keylabel  ALL = 'Total'
             mean = 'Average'
              std = 'Standard Deviation';
run;

/* Eliminating the N column in a PROC TABULATE table
*/

title "Eliminating the 'N' Row from the Table";
proc tabulate data=learn.blood format=6.;
    class Gender;
    table Gender*n=' ';
run;


/* Demonstrating a more complex table
*/

title "Combining CLASS and Analysis Variables";
proc tabulate data=learn.blood format=comma9.2 noseps;
    class Gender AgeGroup;
    var RBC WBC Chol;
    table (Gender=' ' ALL)*(AgeGroup=' ' All), RBC*(n*f=3. mean*f=5.1)
                                               WBC*(n*f=3. mean*f=comma7.)
                                               Chol*(n*f=4. mean*f=7.1);
    keylabel ALL = 'Total';
run;

/* Computing percentages in a one-dimensional table
*/

title "Counts and Percentages";
proc tabulate data=learn.blood format=6.;
     class Blood;
     table Blood*(n pctn);
run;

/* Improving the appearance of output from Program 18-14
*/

proc format;
      picture pctfmt low-high='009.9%';
run;
title "Counts and Percentages";
proc tabulate data=learn.blood;
      class Blood;
      table (Blood ALL)*(n*f=5. pctn*f=pctfmt7.1);
      keylabel    n = 'Count'
               pctn = 'Percent';
run;

/* Counts and percentages in a two-dimensional table
*/

proc format;
      picture pctfmt low-high='009.9%';
run;

title "Counts and Percentages";
proc tabulate data=learn.blood noseps;
      class Gender Blood;
      table (Blood ALL), (Gender ALL)*(n*f=5. pctn*f=pctfmt7.1) /RTS=25;

      keylabel ALL = 'Both Genders'
                 n = 'Count'
              pctn = 'Percent';
run;

/* Using COLPCTN to compute column percentages
*/

title "Percents on Column Dimension";
proc tabulate data=learn.blood noseps;
      class Gender Blood;
      table (Blood ALL='All Blood Types'),(Gender ALL)*(n*f=5. colpctn*f=pctfmt7.1) /RTS=25;

      keylabel     All = 'Both Genders'
                     n = 'Count'
               colpctn = 'Percent';
run;

/* Computing percentages on a numeric value
*/

title "Computing Percentages on a Numerical Value";
proc tabulate data=learn.sales;
      class Region;
      var TotalSales;
      table (Region ALL), TotalSales*(n*f=6. sum*f=dollar8. pctsum*f=pctfmt7.);
      keylabel     ALL = 'All Regions'
                     n = 'Number of Sales'
                   sum = 'Average'
                pctsum = 'Percent';
      label TotalSales = 'Total Sales';
run;

/* Demonstrating the effect of missing values on CLASS variables
*/
data learn.missing;
      input ID A $ B $ C $;
datalines;
1 X Y Z
2 X Y Y
3 Z Z Z
4 X X  
5 Y Z  
6 X    
;
title "The Effect of Missing Values on CLASS variables";
proc tabulate data=learn.missing format=4.;
      class A B C;
      table A ALL,B ALL;
run;

/* Adding the PROC TABULATE procedure option MISSING
*/

title "The Effect of Missing Values on CLASS variables";
proc tabulate data=learn.missing format=4. missing;
     class A B;
     table A ALL,B ALL;
run;

/* Demonstrating the MISSTEXT= TABLES option
*/

title "Demonstrating the MISSTEXT TABLES Option";
proc tabulate data=learn.missing format=7. missing;
     class A B;
     table A ALL,B ALL / misstext='no data';
run;

