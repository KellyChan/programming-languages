/*

PROGRAM NAME: Learning SAS by Example
Chapter 5: Creating Formats and Labels

PROGRAMMER: Kelly Chan
Date Written: Aug 15 2013

*/

libname learn 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example';


data learn.test_scores;

      length ID   $ 3 
             Name $ 15;

      input  ID   $ 
             Score1-Score3;

      label  ID     = 'Student ID'  /* variable label */
             Score1 = 'Math Score'
             Score2 = 'Science Score'
             Score3 = 'English Score';
datalines;
1 90 95 98
2 78 77 75
3 88 91 92
;
run;

proc means data=learn.test_scores;
run;


data learn.survey;
      infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\survey.txt';
	  input ID
	        Gender $
			Age
			Salary
			Q1-Q5;
run;

proc print data=learn.survey;
run;


proc format;

       value $gender 'M' = 'Male'         /* code label */
                     'F' = 'Female'
                     ' ' = 'Not entered'
                   other = 'Miscoded';

       value age  low-29 = 'Less than 30'
                   30-50 = '30 to 50'
                 51-high = '51+';
/*
	   value $likert '1' = 'Strongly disagree'
                     '2' = 'Disagree'
                     '3' = 'No opinion'
                     '4' = 'Agree'
                     '5' = 'Strongly agree';
*/

		value $three '1','2' = 'Disagreement'
                         '3' = 'No opinion'
                     '4','5' = 'Agreement';
run;

title "Data Set SURVEY with Formatted Values";
proc print data=learn.survey;

       id ID;
       var Gender 
           Age 
           Salary 
           Q1-Q5;

       format Gender      $gender.
              Age         age.
              /* Q1-Q5       $likert. */
              Salary      dollar11.2;
run;

proc freq data=learn.survey;
       title "Question Frequencies Using the $three Format";
       tables Q1-Q5;
       format Q1-Q5 $three.;
run;
