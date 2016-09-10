
/*

PROGRAM NAME: demog.sas stored in the data folder.
PURPOSE: 
The program reads in data on height and weight (in inches and pounds, respectively) and computes a body mass index (BMI) for each subject.

PROGRAMMER: Kelly Chan
Date Written: Aug 15 2013

*/

*----------------------------------------------------------------*
| Program Name: DEMOG.SAS stored in the c:\books\learning folder |
| Purpose: The program reads in data on height and weight        |
| (in inches and pounds, respectively) and computes a body       |
| mass index (BMI) for each subject.                             |
*----------------------------------------------------------------*;


/*********************************************************
Program name: demog.sas stored in the c:\books\learning
folder.
Purpose: The program reads in data on height and weight
(in inches and pounds, respectively) and computes a body
mass index (BMI) for each subject.
**********************************************************/


/*********************************************************\
| This is another way to make a fancy-looking comment     |
| using the slash star – star slash form.                |
\*********************************************************/


/* This comment contains a style comment embedded
within another comment. Notice that the first star
slash ends the comment and the remaining portion of
the comment will cause a syntax error */


data demographic;
      infile "G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\mydata.txt";
      input Gender $ 
            Age           /* age is in years */
            Height 
            Weight;

	  BMI = (Weight / 2.2) / (Height*.0254)**2; /* Compute a body mass index (BMI) */
run;


title "Gender Frequencies";
proc freq data=demographic;
      tables Gender;
run;

title "Summary Statistics";
proc means data=demographic;
      var Age 
          Height 
          Weight;
run;
