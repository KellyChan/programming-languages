/*

PROGRAM NAME: Learning SAS by Example
Chapter 12: Working with Character Functions

PROGRAMMER: Kelly Chan
Date Written: Aug 16 2013

*/

libname learn 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\libs';

data sales;
      infile 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\Sales.csv' dsd;
	  input EmpID $	
            Name $	
            Region $ 
            Customer $	
            Item $	
            Quantity	
            UnitCost	
            TotalSales;
run;

data long_names;
      set sales;
      if lengthn(Name) gt 6;
run;
proc print data=long_names;
run;

/* Changing values to uppercase
*/

data mixed;
      input Name $1-14
	        ID   16-19;
datalines;
Daniel Fields 123
Patrice Helms 233
Thomas Chien 998
;

data upper;
      set mixed;
      Name = upcase(Name);
run;
data both;
      merge mixed upper;
      by Name;
run;
proc print data=both;
run;

/* Removing Characters from Strings
*/
data address;
       input Name $1-10 Street $11-30 City $31-45 State $47-48 Zip 50-54;
datalines;
ron coDY   1178 HIGHWAY 480    camp verde    tx 78010
jason Tran 123 lake view drive East Rockaway ny 11518
;

/* Converting multiple blanks to a single blank and demonstrating the PROPCASE function
*/
data standard;
      set address;
      Name = compbl(propcase(Name));
      Street = compbl(propcase(Street));
      City = compbl(propcase(City));
      State = upcase(State);
run;
proc print data=standard;
run;


/* Demonstrating the concatenation functions
*/

title "Demonstrating the Concatenation Functions";
data _null_;
       Length Join Name1-Name4 $ 15;
       First = 'Ron ';
       Last = 'Cody ';
       Join = ':' || First || ':';
       Name1 = First || Last;
       Name2 = cat(First,Last);
       Name3 = cats(First,Last);
       Name4 = catx(' ',First,Last);
       file print;
       put Join= /
           Name1= /
           Name2= /
           Name3= /
           Name4= /;
run;


/* Demonstrating the TRIM, LEFT, and STRIP functions
*/
data blanks;
      String = ' ABC ';
      ***There are 3 leading and 2 trailing blanks in String;
      JoinLeft = ':' || left(String) || ':';
      JoinTrim = ':' || trim(String) || ':';
      JoinStrip = ':' || strip(String) || ':';
run;
proc print data=blanks;
run;

/* Using the COMPRESS function to remove characters from a string
*/

data learn.phone;
      input Phone $1-16;
datalines;
(908)232-4856
210.343.4757
(516) 343 - 9293
9342342345
;
proc print data=phone;
run;

data new_phone;
     length PhoneNumber $ 10;
     set learn.phone;
     PhoneNumber = compress(Phone,' ()-.');
     drop Phone;
run;

data new_phone;
      length PhoneNumber $ 10;
      set learn.phone;
      PhoneNumber = compress(Phone,'kd');
      *Keep only digits;
      drop Phone;
run;
proc print data=new_phone;
run;


/* Demonstrating the FIND and COMPRESS functions
*/
data mixed_nuts;
      input Weight $ Height $;
datalines;
100Kgs. 59in
180lbs 60inches
88kg 150cm.
50KGS 160CM
;

data English;
       set       mixed_nuts(rename=
                                   (Weight = Char_Weight
                                    Height = Char_Height));
       if      find(Char_Weight,'lb','i') then Weight = input(compress(Char_Weight,,'kd'),8.);
       else if find(Char_Weight,'kg','i') then Weight = 2.2*input(compress(Char_Weight,,'kd'),8.);

       if      find(Char_Height,'in','i') then Height = input(compress(Char_Height,,'kd'),8.);
       else if find(Char_Height,'cm','i') then Height = input(compress(Char_Height,,'kd'),8.)/2.54;
       drop Char_:;
run;

proc print data=English;
run;

/* Demonstrating the FINDW function
*/

data look_for_roger;
      input String $40.;
      if findw(String,'Roger') then Match = 'Yes';
      else                          Match = 'No';
datalines;
Will Rogers
Roger Cody
Was roger here?
Was Roger here?
;

/* Demonstrating the ANYDIGIT function
*/

data mixed only_alppha;
    input ID $10.;
	if anydigit(ID) then output mixed;
	else                 output only_alppha;
datalines;
ABc123
XrayMan
142abc
Agent007
Terminator
;

proc print data=mixed;
run;
proc print data=only_alppha;
run;


/* Demonstrating the NOT functions for data cleaning
*/

data cleaning;
      input Subject Letters $ Numerals $ Both $;
datalines;
1 Apple 12345 XYZ123
2 Ice9 123X Abc.123
3 Help! 999 X1Y2Z3
;

title "Data Cleaning Application";
data _null_;
      file print;
      set cleaning;
      if notalpha(trim(Letters))  then put Subject= Letters=;
      if notdigit(trim(Numerals)) then put Subject= Numerals=;
      if notalnum(trim(Both))     then put Subject= Both=;
run;
proc print data=_null_;
run;

/* Using the VERIFY function for data cleaning
*/

data errors valid;
      input ID $ Answer : $5.;
      if verify(Answer,'ABCDE') then output errors;
      else                            output valid;
datalines;
001 AABDE
002 A5BBD
003 12345
;

proc print data=errors;
run;
proc print data=valid;
run;

/* Using the SUBSTR function to extract substrings
*/

data extract;
      input ID : $10. @@;
      length State $ 2 Gender $ 1 Last $ 5;
      State = substr(ID,1,2);
      Number = input(substr(ID,3,2),3.);
      Gender = substr(ID,5,1);
      Last = substr(ID,6);
datalines;
NJ12M99 NY76F4512 TX91M5
;
proc print data=extract;
run;

/* Demonstrating the SCAN function
*/

data original;
       input Name $ 30.;
datalines;
Jeffrey Smith
Ron Cody
Alan Wilson
Alfred E. Newman
;
data first_last;
      set original;
      length First Last $ 15;
      First = scan(Name,1,' ');
      Last = scan(Name,2,' ');
run;
proc print data=first_last;
run;


/* Using the SCAN function to extract the last name
*/

data last;
      set original;
      length LastName $ 15;
      LastName = scan(Name,-1,' ');
run;
proc sort data=last;
      by LastName;
run;

title "Alphabetical list of names";
proc print data=last noobs;
      var Name;
run;

/* Demonstrating the COMPARE function
*/

data diagnosis;
      input Code $10.;
      if compare(Code,'V450','i:') eq 0 then Match = 'Yes';
      else                                   Match = 'No';
datalines;
V450
v450
v450.100
V900
;

proc print data=diagnosis;
run;

/* Clarifying the use of the colon modifier with the COMPARE function
*/

data _null_;
      String1 = 'ABC ';
      String2 = 'ABCXYZ';
      Compare1 = compare(String1,String2,':');
      Compare2 = compare(trim(String1),String2,':');
      put String1= String2= Compare1= Compare2=;
run;

/* Using the SPEDIS function to perform a fuzzy match
*/

data fuzzy;
      input Name $20.;
      Value = spedis(Name,'Friedman');
datalines;
Friedman
Freedman
Xriedman
Freidman
Friedmann
Alfred
FRIEDMAN
;

proc print data=fuzzy;
run;

/* Demonstrating the TRANSLATE function
*/

data trans;
      input Answer : $5.;
      Answer = translate(Answer,'ABCDE','12345');
datalines;
14325
AB123
51492
;
proc print data=trans;
run;

/* Using the TRANWRD function to standardize an address
*/

data address;
       infile datalines dlm=' ,';
       *Blanks or commas are delimiters;
       input #1 Name $30.
             #2 Line1 $40.
             #3 City & $20. State : $2. Zip : $5.;
       Name = tranwrd(Name,'Mr.',' ');
       Name = tranwrd(Name,'Mrs.',' ');
       Name = tranwrd(Name,'Dr.',' ');
       Name = tranwrd(Name,'Ms.',' ');
       Name = left(Name);
       Line1 = tranwrd(Line1,'Street','St.');
       Line1 = tranwrd(Line1,'Road','Rd.');
       Line1 = tranwrd(Line1,'Avenue','Ave.');
datalines;
Dr. Peter Benchley
123 River Road
Oceanside, NY 11518
Mr. Robert Merrill
878 Ocean Avenue
Long Beach, CA 90818
Mrs. Laura Smith
80 Lazy Brook Road
Flemington, NJ 08822
;
proc print data=address;
run;
