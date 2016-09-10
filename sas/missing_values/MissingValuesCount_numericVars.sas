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
 
proc means data=one NMISS N; run;
