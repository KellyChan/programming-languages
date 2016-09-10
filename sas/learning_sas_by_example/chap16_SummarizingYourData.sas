/*

PROGRAM NAME: Learning SAS by Example
Chapter 16: Summarizing Your Data

PROGRAMMER: Kelly Chan
Date Written: Aug 16 2013

*/

libname learn 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\libs';

/* PROC MEANS with all the defaults
*/

title "PROC MEANS With All the Defaults";
proc means data=learn.blood;
run;

/* Adding a VAR statement and requesting specific statistics with PROC MEANS
*/

title "Selected Statistics Using PROC MEANS";
proc means data=learn.blood n nmiss mean median min max maxdec=1;
       var RBC WBC;
run;

/* Adding a BY statement to PROC MEANS
*/

proc sort data=learn.blood out=blood;
       by Gender;
run;
title "Adding a BY Statement to PROC MEANS";
proc means data=blood n nmiss mean median min max maxdec=1;
       by Gender;
       var RBC WBC;
run;

/* Using a CLASS statement with PROC MEANS
*/

title "Using a CLASS Statement with PROC MEANS";
proc means data=learn.blood n nmiss mean median min max maxdec=1;
       class Gender;
       var RBC WBC;
run;


/* Demonstrating the effect of a formatted CLASS variable
*/

proc format;
       value chol_group low -< 200 = 'Low'
                        200 - high = 'High';
run;
proc means data=learn.blood n nmiss mean median maxdec=1;
       title "Using a CLASS Statement with PROC MEANS";
       class Chol;
       format Chol     chol_group.;
       var    RBC WBC;
run;

/* Creating a summary data set using PROC MEANS
*/

proc means data=learn.blood noprint;
       var RBC WBC;
       output out = my_summary mean = MeanRBC MeanWBC;
run;
title "Listing of MY_SUMMARY";
proc print data=my_summary noobs;
run;

/* Outputting more than one statistic with PROC MEANS
*/

proc means data=learn.blood noprint;
       var RBC WBC;
       output out = many_stats mean = M_RBC M_WBC
                                  n = N_RBC N_WBC
                              nmiss = Miss_RBC Miss_WBC
                             median = Med_RBC Med_WBC;
run;
proc print data=many_stats noobs;
run;

/* Demonstrating the OUTPUT option AUTONAME
*/

proc means data=learn.blood noprint;
       var RBC WBC;
       output out = many_stats mean =
                                  n =
                              nmiss =
                             median = / autoname;
run;
proc print data=many_stats noobs;
run;

/* Adding a BY statement to PROC MEANS
*/

proc sort data=learn.blood out=blood;
       by Gender;
run;
proc means data=blood noprint;
       by Gender;
       var RBC WBC;
       output out = by_gender mean = MeanRBC MeanWBC
                                 n = N_RBC N_WBC;
run;
proc print data=by_gender;
run;

/* Adding a CLASS statement to PROC MEANS
*/

proc means data=blood noprint;
       class Gender;
       var RBC WBC;
       output out = with_class mean = MeanRBC MeanWBC
                                  n = N_RBC N_WBC;
run;
proc print data=with_class;
run;

/* Adding the NWAY option to PROC MEANS
*/

proc means data=blood noprint nway;
       class Gender;
       var RBC WBC;
       output out = with_class mean = MeanRBC MeanWBC
                                  n = N_RBC N_WBC;
run;

proc print data=with_class;
run;

/* Using two CLASS variables with PROC MEANS
*/

proc means data=learn.blood noprint;
       class Gender AgeGroup;
       var RBC WBC;
       output out = summary mean =
                               n = / autoname;
run;

proc print data=summary;
run;

/* Adding the CHARTYPE procedure option to PROC MEANS
*/

proc means data=learn.blood noprint chartype;
       class Gender AgeGroup;
       var RBC WBC;
       output out = summary mean =
                               n = / autoname;
run;
proc print data=summary;
run;

/* Using the _TYPE_ variable to select cell means
*/

title "Statistics Broken Down by Gender";
proc print data=summary(drop = _freq_) noobs;
      where _TYPE_ = '10';
run;

/* Using a DATA step to create separate summary data sets
*/

data grand(drop = Gender AgeGroup) by_gender(drop = AgeGroup)
                                       by_age(drop = Gender)
      cellmeans;

set summary;
drop _type_;
rename _freq_ = Number;
if      _type_ = '00' then output grand;
else if _type_ = '01' then output by_age;
else if _type_ = '10' then output by_gender;
else if _type_ = '11' then output cellmeans;
run;

proc print data=grand;
run;
proc print data=by_age;
run;
proc print data=by_gender;
run;
proc print data=cellmeans;
run;

/* Selecting different statistics for each variable using PROC MEANS
*/

proc means data=learn.blood noprint nway;
       class Gender AgeGroup;
       output out = summary(drop = _:) mean(RBC WBC)   =
                                       n(RBC WBC Chol) =
                                       median(Chol)    = / autoname;
run;

proc print data=summary;
run;


