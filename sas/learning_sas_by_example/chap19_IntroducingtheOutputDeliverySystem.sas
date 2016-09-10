/*

PROGRAM NAME: Learning SAS by Example
Chapter 19: Introducing the Output Delivery System

PROGRAMMER: Kelly Chan
Date Written: Aug 16 2013

*/

libname learn 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\libs';
libname myfmts 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\formats';
options fmtsearch=(myfmts);

/* Sending SAS output to an HTML file
*/

ods html file='G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\output\sample.html';
title "Listing of TEST_SCORES";
proc print data=learn.survey;
      title2 "Sample of HTML Output - all defaults";
      id ID;
      var Gender Ques1-Ques5;
run;

title "Descriptive Statistics";
proc means data=learn.survey n mean min max;
     var Ques1-Ques5;
run;
ods html close;


/* Creating a table of contents for HTML output
*/

ods html body = 'body_sample.html'
         contents = 'contents_sample.html'
         frame = 'frame_sample.html'
         path = 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\output' (url=none);
title "Using ODS to Create a Table of Contents";
proc print data=learn.survey;
      id ID;
      var Gender Ques1-Ques5;
run;

title "Descriptive Statistics";
proc means data=learn.survey n mean min max;
      var Ques1-Ques5;
run;
ods html close;

/* Choosing a style for HTML output
*/

ods html file = 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\output\styles.html'
         style=FancyPrinter;
title "Listing of TEST_SCORES";
proc print data=learn.survey;
     id ID;
     var Gender Ques1-Ques5;
run;
ods html close;

/* Using an ODS SELECT statement to restrict PROC UNIVARIATE output
*/

ods select extremeobs;
title "Extreme Values of RBC";
proc Univariate data=learn.blood;
     id ID;
     var RBC;
run;

/* Using the ODS TRACE statement to identify output objects
*/

ods trace on;
title "Extreme Values of RBC";
proc Univariate data=learn.blood;
    id ID;
    var RBC;
run;
ods trace off;

/* Using ODS to send procedure output to a SAS data set
*/

ods listing close;
ods output ttests=t_test_data;
proc ttest data=learn.blood;
    class Gender;
    var RBC WBC Chol;
run;

ods listing;
title "Listing of T_TEST_DATA";
proc print data=t_test_data;
run;

/* Using an output data set to create a simplified report
*/

title "T-Test Results - Using Equal Variance Method";
proc report data=t_test_data nowd headline;
      where Variances = "Equal";
      columns Variable tValue ProbT;
      define Variable /         width=8;
      define tValue / "T-Value" width=7 format=7.2;
      define ProbT / "P-Value"  width=7 format=7.5;
run;

