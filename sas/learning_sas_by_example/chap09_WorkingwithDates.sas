/*

PROGRAM NAME: Learning SAS by Example
Chapter 9: Working with Dates

PROGRAMMER: Kelly Chan
Date Written: Aug 15 2013

*/

libname learn 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\libs';

data four_dates;
      infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\dates.txt' truncover;
      input @1  Subject   $3.
            @5  DOB        mmddyy10.
            @16 VisitDate  mmddyy8.
            @26 TwoDigit   mmddyy8.
            @34 LastDate   date9.;

	  format DOB VisitDate     date9.
             TwoDigit LastDate mmddyy10.;
run;

proc print data=four_dates;
run;

/* Compute a person's age in years
*/

data ages;
      set four_dates;
      Age = yrdif(DOB,VisitDate,'Actual');
run;

title "Listing of AGES";
proc print data=ages;
      id Subject;
      var DOB VisitDate Age;
run;

/* Demonstrating a date constant
*/

data ages;
      set four_dates;
      Age = yrdif(DOB,'01Jan2006'd,'Actual');
run;

title "Listing of AGES";
proc print data=ages;
      id Subject;
      var DOB Age;
      format Age 5.1;
run;


/* Using the TODAY function to return the current date
*/

data ages;
     set four_dates;
     Age = yrdif(DOB,today(),'Actual');
run;
title "Listing of AGES";
proc print data=ages;
      id Subject;
      var DOB Age;
      format Age 5.1;
run;

/* Extracting the day of the week, day of the month, month, and year from a SAS date
*/

data extract;
      set four_dates;
      Day = weekday(DOB);
      DayOfMonth = day(DOB);
      Month = Month(DOB);
      Year = year(DOB);
run;

title "Listing of EXTRACT";
proc print data=extract noobs;
      var DOB Day DayOfMonth Month Year;
run;


/* Using the MDY function to create a SAS date from month, day, and year values
*/

data mdy_example;
      set learn.month_day_year;
      Date = mdy(Month, Day, Year);
      format Date mmddyy10.;
run;
proc print data=mdy_example noobs;
run;

/* Substituting the 15th of the month when a Day value is missing
*/

data substitute;
      set learn.month_day_year;
      if   missing(day) then Date = mdy(Month,15,Year);
      else                   Date = mdy(Month,Day,Year);
      format Date mmddyy10.;
run;

/* Demonstrating the INTCK function
*/

data frequency;
       set learn.hosp(keep=AdmitDate 
                      where=(AdmitDate between '01Jan2003'd and '30Jun2006'd));
       Quarter = intck('qtr','01Jan2003'd,AdmitDate);
run;

title "Frequency of Admissions";
proc freq data=frequency noprint;
      tables Quarter / out=admit_per_quarter;
run;

goptions ftitle=swiss ftext=swiss;
symbol v=dot i=sm color=black width=2;
title height=2 "Frequency of Admissions From";
title2 height=2 "January 1, 2003 and June 30, 2006";

proc gplot data=admit_per_quarter;
       plot Count * Quarter;
run;
quit;

/* Using the INTNX function to compute dates 6 months after discharge
*/

data followup;
      set learn.discharge;
      FollowDate = intnx('month',DischrDate,6);
      format FollowDate date9.;
run;

/* Demonstrating the SAMEDAY alignment with the INTNX function
*/

data followup;
     set learn.discharge;
     FollowDate = intnx('month',DischrDate,6,'sameday');
     format FollowDate date9.;
run;
