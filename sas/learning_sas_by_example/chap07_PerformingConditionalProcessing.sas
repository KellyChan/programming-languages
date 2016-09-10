/*

PROGRAM NAME: Learning SAS by Example
Chapter 7: Performing Conditional Processing

PROGRAMMER: Kelly Chan
Date Written: Aug 15 2013

*/

libname learn 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\libs';

data conditional;
       length Gender $ 1
                Quiz $ 2;
       input Age 
             Gender 
             Midterm 
             Quiz 
             FinalExam;

        /* 
		   Age with missing values

        if Age lt 20               then AgeGroup = 1;
        if Age ge 20 and Age lt 40 then AgeGroup = 2;
        if Age ge 40 and Age lt 60 then AgeGroup = 3;
        if Age ge 60               then AgeGroup = 4;

	    */

		/* Age without missing values 
		if      Age lt 20 and not missing(age) then AgeGroup = 1;
        else if Age ge 20 and Age lt 40        then AgeGroup = 2;
        else if Age ge 40 and Age lt 60        then AgeGroup = 3;
        else if Age ge 60                      then AgeGroup = 4;
	    */

		/* Age without missing values (alternative) */
		if      missing(Age)  then AgeGroup = .;
        else if Age lt 20     then AgeGroup = 1;
        else if Age lt 40     then AgeGroup = 2;
        else if Age lt 60     then AgeGroup = 3;
        else if Age ge 60     then AgeGroup = 4;

		/* Using a SELECT Statement for Logical Tests*/
        select (AgeGroup);
             when (1) Limit = 110;
             when (2) Limit = 120;
             when (3,5) Limit = 130;
             otherwise;
        end;

datalines;
21 M 80 B- 82
. F 90 A 93
35 M 87 B+ 85
48 F . . 76
59 F 95 A+ 97
15 M 88 . 93
67 F 97 A 91
. M 62 F 67
35 F 77 C- 77
49 M 59 C 81
;

title "Listing of CONDITIONAL";
proc print data=conditional noobs;
run;


/* Demonstrating a subsetting IF statement
*/

data females;

      length Gender $ 1
               Quiz $ 2;
      input Age 
            Gender 
            Midterm 
            Quiz 
            FinalExam;

      if Gender eq 'F';

datalines;
21 M 80 B- 82
. F 90 A 93
35 M 87 B+ 85
48 F . . 76
59 F 95 A+ 97
15 M 88 . 93
67 F 97 A 91
. M 62 F 67
35 F 77 C- 77
49 M 59 C 81
;
title "Listing of FEMALES";
proc print data=Females noobs;
run;

/* Combining various Boolean operators
*/

title "Example of Boolan Expressions";
proc print data=conditional;
         where Gender eq 'M' and  (Quiz in ('B-' 'A+')  or  Midterm gt 80);
		 id Gender;
         var Gender Quiz Age Midterm AgeGroup;
run;

data believe_it_or_not;
      input X;
      if    X = 3 or 4 then Match = 'Yes';  /* whatever X, it is true, boolean */
      else                  Match = 'No';
datalines;
3
7
.
;
title "Listing of BELIEVE_IT_OR_NOT";
proc print data=believe_it_or_not noobs;
run;

/* Using a WHERE statement to subset a SAS data set
*/
data females;
      set conditional;
      where Gender eq 'F';
run;
proc print data=females;
run;
