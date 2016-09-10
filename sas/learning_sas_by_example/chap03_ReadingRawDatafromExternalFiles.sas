
/*

PROGRAM NAME: Learning SAS by Example
Chapter 3: Reading Raw Data from External Files

PROGRAMMER: Kelly Chan
Date Written: Aug 15 2013

*/

/* 1. Reading Data Values Separated by Blanks
*/
data demographics;
      infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\demo.txt';
      input Gender $ 
            Age 
            Height 
            Weight;
run;


/* 2. Reading Data from csv
*/

filename preston 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\demo.csv';
data demographics;
       infile preston dsd;
       input Gender $ 
             Age 
             Height 
             Weight;
run;

/* 3. Reading Data with ":"
*/

filename test 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\test.txt';
data test;
       infile test dlm=':';
       input Gender $ 
             Age 
             Height 
             Weight;
run;

/* 4. Reading Data with datalines 
*/
data demographic;
       input Gender $ Age Height Weight;
datalines;
M 50 68 155
F 23 60 101
M 65 72 220
F 35 65 133
M 15 71 166
;
run;

/* 5. Reading Data without blanks
*/
data financial;
      infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\financial.txt';
      input Subj    $ 1-3
            DOB     $ 4-13
            Gender  $ 14
            Balance   15-21;
run;

/* 6. Demonstrating formatted input
*/
data financial;
      infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\financial.txt';
      input @1  Subj    $3.
            @4  DOB     mmddyy10.
            @14 Gender  $1.
            @15 Balance  7.;
run;

/* 7. Using informats with list input (csv file)
*/

data list_example;
       infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\list.csv' dsd;
       input  Subj   : $3.
              Name   : $20.
              DOB    : mmddyy10.
              Salary : dollar8.;
       format DOB      date9. 
              Salary   dollar8.;
run;

/* 8. Supplying an INFORMAT statement with list input
*/

data list_example;
       informat Subj $3.
                Name $20.
                DOB mmddyy10.
                Salary dollar8.;
       infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\list.csv' dsd;
       input Subj
             Name
             DOB
             Salary;
       format DOB date9. 
              Salary dollar8.;
run;

/* Demonstrating a FORMAT statement
*/
title "Listing of FINANCIAL";
proc print data=financial;
       format DOB     mmddyy10.
              Balance dollar11.2;
run;

/* Rerunning Program 3-9 with a different format
*/

title "Listing of FINANCIAL";
proc print data=financial;
      format DOB     date9.
             Balance dollar11.2;
run;

title "Listing of data set DEMOGRAPHICS";
proc print data=demographics;
run;
