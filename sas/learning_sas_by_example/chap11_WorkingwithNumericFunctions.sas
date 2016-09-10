/*

PROGRAM NAME: Learning SAS by Example
Chapter 11: Working with Numeric Functions

PROGRAMMER: Kelly Chan
Date Written: Aug 15 2013

*/

libname learn 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\libs';
data truncate;
      input Age Weight;
      Age = int(Age);
      WtKg = round(2.2*Weight,.1);
      Weight = round(Weight);
datalines;
18.8 100.7
25.12 122.4
64.99 188
;
proc print data = truncate;
run;

data test_miss;
set learn.blood;
if Gender = ' ' then MissGender + 1;
if WBC = . then MissWBC + 1;
if RBC = . then MissWBC + 1;
if Chol lt 200 and Chol ne . then Level = 'Low ';
else if Chol ge 200 then Level = 'High';
run;

data learn.blood;
      infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\blood.txt';
      input ID Gender $ Blood $ AgeGroup $ WBC RBC Chol;
run;

data test_miss;
      set learn.blood;
      if      Gender = ' '              then MissGender + 1;
      if      WBC = .                   then MissWBC + 1;
      if      RBC = .                   then MissRBC + 1;
      if      Chol lt 200 and Chol ne . then Level = 'Low ';
      else if Chol ge 200               then Level = 'High';
run;

data test_miss;
      set learn.blood;
      if missing(Gender) then MissGender + 1;
      if missing(WBC)    then MissWBC + 1;
      if missing(RBC)    then MissRBC + 1;
      if      Chol lt 200 and not missing(Chol) then Level = 'Low ';
      else if Chol ge 200                       then Level = 'High';
run;

proc print data=test_miss;
run;

/* Demonstrating the N, MEAN, MIN, and MAX functions
*/

data psych;
      input ID $ Q1-Q10;
      if n(of Q1-Q10) ge 7 then Score = mean(of Q1-Q10);
      MaxScore = max(of Q1-Q10);
      MinScore = min(of Q1-Q10);
datalines;
001 4 1 3 9 1 2 3 5 . 3
002 3 5 4 2 . . . 2 4 .
003 9 8 7 6 5 4 3 2 1 5
;

proc print data=psych;
run;

/* Finding the sum of the three largest values in a list of variables
*/

data three_large;
      set psych(keep=ID Q1-Q10);
      SumThree = sum(largest(1,of Q1-Q10),
                     largest(2,of Q1-Q10),
                     largest(3,of Q1-Q10));
	  Total = sum(0, of Q1-Q10);
run;
proc print data=three_large;
run;

/* Demonstrating the ABS, SQRT, EXP, and LOG functions
*/

data math;
      input x @@;
      Absolute = abs(x);
      Square   = sqrt(x);
      Exponent = exp(x);
      Natural  = log(x);
datalines;
2 -2 10 100
;
proc print data=math;
run;


/* Computing some useful constants with the CONSTANT function
*/

data constance;
      Pi = constant('pi');
      e = constant('e');
      Integer3 = constant('exactint',3);
      Integer4 = constant('exactint',4);
      Integer5 = constant('exactint',5);
      Integer6 = constant('exactint',6);
      Integer7 = constant('exactint',7);
      Integer8 = constant('exactint',8);
run;
proc print data=constance;
run;

/* Using the RANUNI function to randomly select observations
*/

data subset;
      set learn.blood;
      if ranuni(1347564) le .1;
run;
proc print data=subset;
run;

/* Using PROC SURVEYSELECT to obtain a random sample
*/

proc surveyselect data=learn.blood
                    out=subset
                    method=srs
                    sampsize=100;
run;

/* Special Functions
*/

data CHARS;
      input Height $ Weight $ Date $;
datalines;
58 155 10/21/1950
63 200 5/6/2005
45 79 11/12/2004
;

/* Using the INPUT function to perform a character-to-numericm conversion
*/

data nums;
     set      CHARS (rename=
                             (Height = Char_Height
                              Weight = Char_Weight
                              Date = Char_Date));
     Height = input(Char_Height,8.);
     Weight = input(Char_Weight,8.);
     Date = input(Char_Date,mmddyy10.);
     drop Char_Height Char_Weight Char_Date;
run;
proc print data=nums;
run;

/* Demonstrating the PUT function
*/

proc format;
      value agefmt low-<20 = 'Group One'
                    20-<40 = 'Group Two'
                   40-high = 'Group Three';
run;
data convert;
      set learn.numeric;
      Char_Date = put(Date,date9.);
      AgeGroup = put(Age,agefmt.);
      Char_Cost = put(Cost,dollar10.);
      drop Date Cost;
run;

/* Demonstrating the LAG and LAGn functions
*/

data look_back;
      input Time Temperature;
      Prev_temp = lag(Temperature);
      Two_back = lag2(Temperature);
datalines;
1 60
2 62
3 65
4 70
;
proc print data=look_back;
run;

/* Demonstrating what happens when you execute a LAG function conditionally
*/

data laggard;
      input x @@;
      if X ge 5 then Last_x = lag(x);
datalines;
9 8 7 1 2 12
;
proc print data=laggard;
run;

/* Using the LAG function to compute interobservation differences
*/

data diff;
     input Time Temperature;
     Diff_temp = Temperature - lag(Temperature);
datalines;
1 60
2 62
 65
4 70
;

/* Demonstrating the DIF function
*/

data diff;
     input Time Temperature;
     Diff_temp = dif(Temperature);
datalines;
1 60
2 62
3 65
4 70
;

proc print data=diff;
run;
