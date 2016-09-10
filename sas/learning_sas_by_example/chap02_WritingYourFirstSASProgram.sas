
data demographic;
infile "G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\data\mydata.txt";
input Gender $ 
      Age 
      Height 
      Weight;
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
