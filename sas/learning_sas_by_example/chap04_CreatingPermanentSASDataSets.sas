/*

PROGRAM NAME: Learning SAS by Example
Chapter 4: Creating Permanent SAS Data Sets

PROGRAMMER: Kelly Chan
Date Written: Aug 15 2013

*/

libname cha4 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example';

data cha4.test_scores;

        length ID   $ 3 
               Name $ 15;
        input  ID $ 
               Score1-Score3 
               Name $;
datalines;
1 90 95 98
2 78 77 75
3 88 91 92
;
run;

proc print data=cha4.test_scores;
run;

title "The Descriptor Portion of Data Set TEST_SCORES";
proc contents data=cha4.test_scores;
run;

title "The Descriptor Portion of Data Set TEST_SCORES";
proc contents data=cha4.test_scores varnum;  /* varnum --> Variables in Creation Order */
run;

/* Using a LIBNAME in a new SAS session
*/
libname proj99 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example';
title "Descriptor Portion of Data Set TEST_SCORES";
proc contents data=proj99.test_scores varnum;
run;

/* Using PROC CONTENTS to list the names of all the SAS data sets in a SAS library
*/
title "Listing All the SAS Data Sets in a Library";
proc contents data=cha4._all_ nods; /* list data sets in SAS library*/
run;

/* Using observations from a SAS data set as input to a new SAS data set
*/

data new;
     set cha4.test_scores;
     AveScore = mean(of score1-score3);
run;

title "Listing of Data Set NEW";
proc print data=new;
     var ID Score1-Score3 AveScore;
run;

/* Demonstrating a DATA _NULL_ step
*/

data _null_;
      set cha4.test_scores;
      if score1 ge 95 or score2 ge 95 or score3 ge 95 then
           put ID= Score1= Score2= Score3=;
run;

proc print _null_;
run;
