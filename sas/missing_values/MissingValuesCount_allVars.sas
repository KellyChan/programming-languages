/* create sample data */
data one;
  input a $ b $ c $ d e;
cards;
a . a 1 3
. b . 2 4
a a a . 5
. . b 3 5
a a a . 6
a a a . 7
a a a 2 8
;
run;
 
/* create a format to group missing and nonmissing */
proc format;
 value $missfmt ' '='Missing' other='Not Missing';
 value  missfmt  . ='Missing' other='Not Missing';
run;
 
proc freq data=one; 
format _CHAR_ $missfmt.; /* apply format for the duration of this PROC */
tables _CHAR_ / missing missprint nocum nopercent;
format _NUMERIC_ missfmt.;
tables _NUMERIC_ / missing missprint nocum nopercent;
run;
