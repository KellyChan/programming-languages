/*

PROGRAM NAME: Learning SAS by Example
Chapter 5: Creating Formats and Labels

PROGRAMMER: Kelly Chan
Date Written: Aug 15 2013

*/

libname learn 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\libs';
libname myfmts 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\formats';
options fmtsearch=(myfmts);

proc format library=myfmts;

              value $gender 'M' = 'Male'
                            'F' = 'Female'
                            ' ' = 'Not entered'
                          other = 'Miscoded';

               value age low-29 = 'Less than 30'
                          30-50 = '30 to 50'
                        51-high = '51+';

               value $likert '1' = 'Strongly disagree'
                             '2' = 'Disagree'
                             '3' = 'No opinion'
                             '4' = 'Agree'
                             '5' = 'Strongly agree';
run;

data learn.survey;

        infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\survey.txt' pad;
        input ID            : $3.
              Gender        : $1.
              Age
              Salary
              (Ques1-Ques5) (: $1.);

        format Gender      $gender.
               Age          age.
               Ques1-Ques5 $likert.
               Salary       dollar10.0;

        label ID = 'Subject ID'
              Gender = 'Gender'
              Age = 'Age as of 1/1/2006'
              Salary = 'Yearly Salary'
              Ques1 = 'The governor is doing a good job?'
              Ques2 = 'The property tax should be lowered'
              Ques3 = 'Guns should be banned'
              Ques4 = 'Expand the Green Acre program'
              Ques5 = 'The school needs to be expanded';
run;


title "Data set SURVEY";
proc contents data=learn.survey varnum;
run;

/* Displaying format definitions in a user-created library
*/

title "Format Definitions in the MYFMTS Library";
proc format library=myfmts fmtlib;
run;

/* Demonstrating a SELECT statement with PROC FORMAT
*/
proc format library=myfmts;
     select age $likert;
run;
