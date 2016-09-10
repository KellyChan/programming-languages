/*

PROGRAM NAME: Learning SAS by Example
Chapter 8: Performing Conditional Processing

PROGRAMMER: Kelly Chan
Date Written: Aug 15 2013

*/

libname learn 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\libs';

data grades;

     length Gender $ 1
              Quiz $ 2
            AgeGrp $ 13;

     infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\grades.txt' missover;
     input Age 
           Gender 
           Midterm 
           Quiz 
           FinalExam;
      
	/*  if statment
     if missing(Age) then delete;
     if Age le 39 then Agegrp = 'Younger group';
     if Age le 39 then Grade = .4*Midterm + .6*FinalExam;
     if Age gt 39 then Agegrp = 'Older group';
     if Age gt 39 then Grade = (Midterm + FinalExam)/2;
     */

     /* if statment with do*/
	 if missing(Age) then delete;
     if Age le 39 then do;
           Agegrp = 'Younger group';
            Grade = .4*Midterm + .6*FinalExam;
     end;
     else if Age gt 39 then do;
           Agegrp = 'Older group';
            Grade = (Midterm + FinalExam)/2;
     end;
run;

title "Listing of GRADES";
proc print data=grades noobs;
run;


data revenue;

       retain Total 0;

       input     Day : $3.
             Revenue : dollar6.;

               Total = Total + Revenue; /* Note: this does not work */

       format Revenue Total dollar8.;

datalines;
Mon $1,000
Tue $1,500
Wed .
Thu $2,000
Fri $3,000
;


data revenue;
      input Day : $3.
            Revenue : dollar6.;
      Total + Revenue;
      format Revenue Total dollar8.;
datalines;
Mon $1,000
Tue $1,500
Wed .
Thu $2,000
Fri $3,000
;

proc print data=revenue;
run;

data test;
      input x;
      if missing(x) then MissCounter + 1;
datalines;
2
.
7
.
;
proc print data=test;
run;


data compound;
       Interest = .0375;
          Total = 100;

       Year + 1;
       Total + Interest*Total;
       output;

       Year + 1;
       Total + Interest*Total;
       output;

       Year + 1;
       Total + Interest*Total;
       output;

       format Total dollar10.2;
run;


data compound;
       Interest = .0375;
          Total = 100;

       do Year = 1 to 3;
           Total + Interest*Total;
           output;
       end;

       format Total dollar10.2;
run;

title "Listing of COMPOUND";
proc print data=compound noobs;
run;


data table;
      do n = 1 to 10;
          Square = n*n;
          SquareRoot = sqrt(n);
          output;
      end;
run;

title "Table of Squares and Square Roots";
proc print data=table noobs;
run;


data equation;
      do X = -10 to 10 by .01;
            Y = 2*x**3 - 5*X**2 + 15*X -8;
            output;
      end;
run;

symbol value=none interpol=sm width=2;
title "Plot of Y versus X";
proc gplot data=equation;
       plot Y * X;
run;


data easyway;
       do Group = 'Placebo','Active';
            do Subj = 1 to 5;
                  input Score @;
                  output;
            end;
       end;
datalines;
250 222 230 210 199
166 183 123 129 234
;
proc print data=easyway;
run;


data double;
      Interest = .0375;
         Total = 100;

      do until (Total ge 200);
           Year + 1;
           Total = Total + Interest*Total;
           output;
      end;

      format Total dollar10.2;
run;

data double;
      Interest = .0375;
         Total = 300;

      do until (Total gt 200);
         Year + 1;
         Total = Total + Interest*Total;
         output;
      end;
      format Total dollar10.2;
run;

data double;
      Interest = .0375;
         Total = 100;

      do while (Total le 200);
         Year + 1;
         Total = Total + Interest*Total;
         output;
      end;
      format Total dollar10.2;
run;


data double;
     Interest = .0375;
     Total = 300;

     do while (Total lt 200);
           Year + 1;
           Total = Total + Interest*Total;
           output;
     end;

     format Total dollar10.2;
run;

data double;
      Interest = .0375;
         Total = 100;

      do Year = 1 to 100 until (Total gt 200);
             Total = Total + Interest*Total;
             output;
      end;

      format Total dollar10.2;
run;

title "Listing of DOUBLE";
proc print data=double noobs;
run;

data leave_it;
      Interest = .0375;
         Total = 100;

      do Year = 1 to 100;
          Total = Total + Interest*Total;
          output;
          if Total ge 200 then leave;
      end;

      format Total dollar10.2;
run;

proc print data=leave_it;
run;


data continue_on;
        Interest = .0375;
           Total = 100;

        do Year = 1 to 100 until (Total ge 200);
           Total = Total + Interest*Total;
           if Total le 150 then continue;
           output;
        end;

        format Total dollar10.2;
run;
proc print data=continue_on;
run;
