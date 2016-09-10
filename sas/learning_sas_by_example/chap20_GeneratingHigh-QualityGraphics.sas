/*

PROGRAM NAME: Learning SAS by Example
Chapter 20: Generating High-Quality Graphics

PROGRAMMER: Kelly Chan
Date Written: Aug 16 2013

*/

libname learn 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\libs';
libname myfmts 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\formats';
options fmtsearch=(myfmts);

/* Producing a simple bar chart using PROC GCHART
*/

title "Distribution of Blood Types";
pattern value=empty;
proc gchart data=learn.blood;
      vbar Blood;
run;
quit;

/* Creating a simple pie chart
*/

title "Creating a Pie Chart";
pattern value=pempty;
proc gchart data=learn.blood;
      pie Blood / noheading;
run;
quit;

/* Creating a bar chart for a continuous variable
*/

pattern value=empty;
proc gchart data=learn.blood;
     vbar WBC;
run;

/* Selecting your own midpoints for the chart
*/

pattern value=L2;
title "Distribution of WBC's";
proc gchart data=learn.blood;
      vbar WBC / midpoints=4000 to 11000 by 1000;
      format WBC comma6.;
run;

/* Demonstrating the need for the DISCRETE option of PROC GCHART
*/

data day_of_week;
     set learn.hosp;
     Day = weekday(AdmitDate);
run;
title "Visits by Day of Week";
pattern value=R1;
proc gchart data=day_of_week;
       vbar Day;
run;

/* Demonstrating the DISCRETE option of PROC GCHART
*/

title "Visits by Month of the Year";
pattern value=R1;
proc gchart data=day_of_week;
      vbar Day / discrete;
run;

/* Creating a bar chart where the height of the bars represents sums
*/

title "Total Sales by Region";
pattern1 value=L1;
axis1 order=('North' 'South' 'East' 'West');
proc gchart data=learn.sales;
     vbar Region / sumvar=TotalSales
                   type=sum
                   maxis=axis1;
     format TotalSales dollar8.;
run;
quit;


/* Creating Bar Charts Representing Means
*/

title "Average Cholesterol by Gender";
pattern1 value=L1;
proc gchart data=learn.blood;
       vbar Gender / sumvar=Chol
                     type=mean;
run;
quit;

/* Adding Another Variable to the Chart
*/

title "Average Cholesterol by Gender";
pattern1 value=L1;
proc gchart data=learn.blood;
      vbar Gender / sumvar=Chol
                    type=mean
                    group=Blood;
run;
quit;

/* Demonstrating the SUBGROUP= option
*/

title "Average Cholesterol by Gender";
pattern1 value=L1;
pattern2 value=R3;
proc gchart data=learn.blood;
       vbar Blood / subgroup=Gender;
run;
quit;

/* Creating a simple scatter plot using all the defaults
*/

title "Scatter Plot of RBC by WBC";
proc gplot data=learn.blood;
      plot RBC * WBC;
run;

/* Changing the plotting symbol and controlling the axis ranges
*/

title "Scatter Plot of RBC by WBC";
symbol value=dot;
proc gplot data=learn.blood;
    plot RBC * WBC / haxis=4000 to 10000 by 1000
                     vaxis=1 to 10 by 1;
run;

/* Joining the points with straight lines (first attempt)
*/

title "Scatter Plot of RBC by WBC";
title2 h=1.2 "Interpolation Methods";
symbol value=dot interpol=join width=2;
proc gplot data=learn.blood;
     plot RBC * WBC;
run;


/* Using the JOIN option on a sorted data set
*/

proc sort data=learn.blood out=blood;
    by WBC;
run;
title "Scatter Plot of RBC by WBC";
title2 h=1.2 "Interpolation Methods";
symbol value=dot interpol=join width=2;
proc gplot data=blood;
      plot RBC * WBC;
run;

/* Drawing a smooth line through your data points
*/

title "Scatter Plot of RBC by WBC";
title2 h=1.2 "Interprelation Methods";
symbol value=dot interpol=sms line=1 width=2;
proc gplot data=learn.blood;
       plot RBC * WBC;
run;
