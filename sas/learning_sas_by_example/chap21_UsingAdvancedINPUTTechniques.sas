/*

PROGRAM NAME: Learning SAS by Example
Chapter 21: Using Advanced INPUT Techniques

PROGRAMMER: Kelly Chan
Date Written: Aug 16 2013

*/

libname learn 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\libs';
libname myfmts 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\formats';
options fmtsearch=(myfmts);

/* Missing values at the end of a line with list input
*/
data missing;
     infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\missing.txt' missover;
     input x y z;
run;
proc print data=missing;
run;

/* Missing values at the end of a line with list input
*/

data short;
      infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\short.txt';
      input Subject $ 1-3
            Name $ 4-19
            Quiz1 20-22
            Quiz2 23-25
            Quiz3 26-28;
run;
proc print data=short;
run;

/* Demonstrating the INFILE PAD option
*/

data short;
      infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\short.txt' pad;
      input Subject $ 1-3
            Name $ 4-19
            Quiz1 20-22
            Quiz2 23-25
            Quiz3 26-28;
run;
proc print data=short;
run;

/* Demonstrating the END= option in the INFILE statement
*/

data _null_;
     file print;
     infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\month.txt' end=Last;
     input @1 Month $3.
           @5 MonthTotal 4.;
     YearTotal + MonthTotal;
     if last then put "Total for the year is" YearTotal dollar8.;
run;

/* Demonstrating the OBS= INFILE option to read the first three lines of data
*/

data readthree;
      infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\month.txt' obs=3;
      input @1 Month $3.
            @5 MonthTotal 4.;
run;
proc print data=readthree;
run;

/* Using the OBS= and FIRSTOBS= INFILE options together
*/

data read5to7;
      infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\month.txt'  firstobs=5 obs=7;
      input @1 Month      $3.
            @5 MonthTotal  4.;
run;
proc print data=read5to7;
run;

/* Using the END= option to read data from multiple files
*/

data combined;
    if   finished = 0 then infile 'alpha.txt' end=finished;
    else                   infile 'beta.txt';
    input ...;
run;

/* Reading Data from Multiple Files Using a FILENAME Statement
*/

filename bilbo ('alpha.txt' 'beta.txt');

data combined;
      infile bilbo;
      input ...;


/* Reading external filenames from an external file
*/

data readmany;
     infile 'filenames.txt';
     input ExternalNames $ 40.;
     infile dummy filevar=ExternalNames end=Last;

     do until (last);
          input . . .;
          output;
     end;
run;

/* Reading external filenames using a DATALINES statement
*/

data readmany;
      input ExternalNames $ 40.;
      infile dummy filevar=ExternalNames end=Last;
      do until (last);
          input . . .;
          output;
     end;
datalines;
c:\books\learning\data1.txt
c:\books\learning\moredata.txt
c:\books\learning\fred.txt
;

/* Reading multiple lines of data to create one observation
*/

data health;
       infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\health.txt' pad;
       input #1 @1  Subj $3.
                @4  DOB mmddyy10.
                @14 Weight 3.
             #2 @4 HR 3.
                @7 SBP 3.
                @10 DBP 3.;
       format DOB mmddyy10.;
run;

/* Using an alternate method of reading multiple lines of data to form one SAS observation
*/

data health;
      infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\health.txt' pad;
      input @1 Subj $3.
            @4 DOB mmddyy10.
            @14 Weight 3. /
            @4 HR 3.
            @7 SBP 3.
            @10 DBP 3.;
      format DOB mmddyy10.;
run;
proc print data=health;
run;

/* Incorrect attempt to read a file of mixed record types
*/

data survey;
     infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\survey56.txt' pad;
     input @9 year $4. @;
     if      year = '2005' then input @1 Number
                                      @4 Q1
                                      @5 Q2
                                      @6 Q3
                                      @7 Q4;
     else if year = '2006' then input @1 Number
                                      @4 Q1
                                      @5 Q2
                                      @6 Q2B
                                      @7 Q3
                                      @8 Q4;
run;
proc print data=survey;
run;

