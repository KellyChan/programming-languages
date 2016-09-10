/*

PROGRAM NAME: Learning SAS by Example
Chapter 17: Counting Frequencies

PROGRAMMER: Kelly Chan
Date Written: Aug 16 2013

*/

libname learn 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\libs';
libname myfmts 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\formats';
options fmtsearch=(myfmts);

/* Counting frequencies: one-way tables using PROC FREQ
*/

title "PROC FREQ with all the Defaults";
proc freq data=learn.survey;
run;

/* Adding a TABLES statement to PROC FREQ
*/

title "Adding a TABLES Statement";
proc freq data=learn.survey;
       tables Gender Ques1-Ques3 / nocum;
run;

/* Adding formats to Program 17-2
*/

title "Adding Formats";
proc freq data=learn.survey;
       tables Gender Ques1-Ques3 / nocum;
       format Gender $gender.
              Ques1-Ques3 $likert.;
run;

/* Using formats to group values
*/

proc format;
       value agegroup
                              low-<30 = 'Less than 30'
                               30-<60 = '30 to 59'
                              60-high = '60 and higher';
       value $agree_disagree
                              '1','2' = 'Generally disagree'
                                  '3' = 'No opinion'
                              '4','5' = 'Generally agree';
run;

title "Using Formats to Create Groups";
proc freq data=learn.survey;
       tables Age Ques5 / nocum nopercent;
       format Age   agegroup.
              Ques5 $agree_disagree.;
run;

/* Demonstrating a problem in how PROC FREQ groups values
*/
data learn.grouping;
      input X;
datalines;
2
2
3
3
4
4
4
.
5
5
5
6
;
proc format;
           value two
                      low-3 = 'Group 1'
                        4-5 = 'Group 2'
						.   = 'Missing'
                      other = 'Other values';
run;
title "Grouping Values (First Try)";
proc freq data=learn.grouping;
       tables X / nocum nopercent;
       format X two.;
run;

/* Demonstrating the effect of the MISSING option of PROC FREQ
*/

title "PROC FREQ Using the MISSING Option";
proc freq data=learn.grouping;
      tables X / missing;
      format X two.;
run;

title "PROC FREQ Without the MISSING Option";
proc freq data=learn.grouping;
      format X two.;
      tables X;
run;

/* Demonstrating the ORDER= option of PROC FREQ
*/

proc format;
        value darwin
                      1 = 'Yellow'
                      2 = 'Blue'
                      3 = 'Red'
                      4 = 'Green'
                      . = 'Missing';
run;

data test;
        input Color @@;
datalines;
3 4 1 2 3 3 3 1 2 2
;

title "Default Order (Internal)";
proc freq data=test;
       tables Color / nocum nopercent missing;
       format Color darwin.;
run;

/* Demonstrating the ORDER= formatted, data, and freq options
*/

title "ORDER = formatted";
proc freq data=test order=formatted;
     tables Color / nocum nopercent;
     format Color darwin.;
run;

title "ORDER = data";
proc freq data=test order=data;
     tables Color / nocum nopercent;
     format Color darwin.;
run;

title "ORDER = freq";
proc freq data=test order=freq;
     tables Color / nocum nopercent;
     format Color darwin.;
run;

/* Requesting a two-way table
*/

title "A Two-way Table of Gender by Blood Type";
proc freq data=learn.blood;
      tables Gender * Blood;
run;

/* Requesting a three-way table with PROC FREQ
*/

title "Example of a Three-way Table";
proc freq data=learn.blood;
     tables Gender * AgeGroup * Blood / nocol norow nopercent;
run;


