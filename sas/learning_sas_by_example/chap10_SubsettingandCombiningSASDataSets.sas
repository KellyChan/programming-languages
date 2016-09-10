/*

PROGRAM NAME: Learning SAS by Example
Chapter 10: Subsetting and Combining SAS Data Sets

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

data females;
      set learn.survey;
      where Gender = 'F';
run;

data females;
      set learn.survey(drop=Salary);
      where gender = 'F';
run;

/* data daparted by males and females
*/
data males females;
      set learn.survey;
      if      Gender = 'F' then output females;
      else if Gender = 'M' then output males;
run;

proc print data=males;
run;
proc print data=females;
run;

/* Adding Observations to a SAS Data Set
*/

data one;
     input rowID ID Name $ Weight;
datalines;
1 7 Adams 210
2 1 Smith 190
3 2 Schneider 110
4 4 Gregory 90
;

data two;
     input rowID ID Name $ Weight;
datalines;
1 9 Shea 120
2 3 O'Brien 180
3 5 Bessler 207
;

data three;
      input rowID ID Gender $ Name $;
datalines;
1 10 M Horvath
2 15 F Stevens
3 20 M Brown
;

data one_two;
     set one two;
run;
proc print data=one_two;
run;

data one_three;
      set one three;
run;
proc print data=one_three;
run;

proc sort data=one;
     by ID;
run;
proc sort data=two;
     by ID;
run;
data interleave;
     set one two;
     by ID;
run;
proc print data=interleave;
run;


/* Combining Detail and Summary Data
*/

data learn.blood;
       infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\blood.txt' pad;
	   input ID Gender $ Blood $ AgeGroup $ Q1 Q2 Chol;
run;

proc means data=learn.blood noprint;
     var Chol;
     output out = means(keep=AveChol)
     mean = AveChol;
run;

data percent;
      set                 learn.blood(keep=Chol);
      if _n_ = 1 then set means;

      PerChol = Chol / AveChol;
      format PerChol percent8.;
run;
proc print data=percent;
run;

/* Merging Two Data Sets
*/

data learn.employee;
       infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\employee.txt' pad;
	   input ID Name $;
run;

data learn.hours;
       infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\hours.txt' pad;
	   input ID Class $ Hours;
run;

proc sort data=employee;
     by ID;
run;
proc sort data=hours;
     by ID;
run;
data combine;
     merge learn.employee learn.hours;
     by ID;
run;

proc print data=combine;
run;

/* Controlling Observations in a Merged Data Set
*/

data new;
     merge learn.employee(in=InEmploy)
           learn.hours (in=InHours);
     by ID;

     file print;
     put ID= InEmploy= InHours= Name= JobClass= Hours=;
run;

/* Using IN= variables to select IDs that are in both data sets
*/

data combine;
      merge learn.employee(in=InEmploy)
            learn.hours(in=InHours);
      by ID;
      if InEmploy and InHours;
run;
proc print data = combine;
run;

/* More Uses for IN= Variables
*/

data in_both
        missing_name(drop = Name);
        merge learn.employee(in=InEmploy)
              learn.hours(in=InHours);
        by ID;

        if      InEmploy and InHours     then output in_both;
        else if InHours and not InEmploy then output missing_name;
run;
proc print data=in_both;
run;
proc print data=missing_name;
run;

/* Demonstrating when a DATA step ends
*/

data short;
       input x;
datalines;
1
2
;
data long;
       input x;
datalines;
3
4
5
6
;

data new;
      set short;
      output;
      set long;
      output;
run;
proc print data=new;
run;

/* Merging Two Data Sets with Different BY Variable Names
*/

data bert;
     input ID X;
datalines;
123 90
222 95
333 100
;

data ernie;
     input EmpNo Y;
datalines;
123 200
222 205
333 317
;

data sesame;
      merge bert
            ernie(rename=(EmpNo = ID));
      by ID;
run;
proc print data=sesame;
run;

/* Merging Two Data Sets with Different BY Variable Data Types
*/

data division1;
      input SS  DOB Gender $;
datalines;
111223333 11/14/1956 M
123456789 05/17/1946 F
987654321 04/01/1977 F
;

data division2;
       input SS $ JobCode $ Salary;
datalines;
111-22-3333 A10 45123
123-45-6789 B5 35400
987-65-4321 A20 87900
;

data division1c;
      set division1(rename=(SS = NumSS));
      SS = put(NumSS, ssn11.);
      drop NumSS;
run;


data division2n;
      set division2(rename=(SS = CharSS));
      SS = input(compress(CharSS,'-'),9.);
      ***Alternative: SS = input(CharSS,comma11.);
      drop CharSS;
run;

data both_divisions;
      ***Note: Both data sets already in order of BY variable;
      merge division1c division2;
      by SS;
run;

/* Updating a Master File from a Transaction File
*/

data prices;
      input ItemCode 1-3
            Description $5-22 Price 24-28;
datalines;
150 50 foot hose      19.95
175 75 foot hose      29.95
200 greeting card     1.99
204 25 lb. grass seed 18.88
208 40 lb. fertilizer 17.98
;

data NEW15DEC2005;
       input ItemCode Price;
datalines;
204 17.87
175 25.11
208 .
;
proc sort data=prices;
      by ItemCode;
run;
proc sort data=new15dec2005;
      by ItemCode;
run;
data prices_15dec2005;
      update prices new15dec2005;
      by ItemCode;
run;

proc print data = prices_15dec2005;
run;
