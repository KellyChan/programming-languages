/*

PROGRAM NAME: Learning SAS by Example
Chapter 14: Displaying Your Data

PROGRAMMER: Kelly Chan
Date Written: Aug 16 2013

*/

libname learn 'G:\eclipseWorkspace\SAS\examples\Learning SAS by Example\libs';

data learn.medical;
	  input @1  Patno      
            @5  Clinic    $11.
            @17 VisitDate  mmddyy10.
            @27 Weight     
            @31 HR         
            @34 DX
            @38 Comment   $60.;
      format VisitDate date9.;
datalines;
001 Mayo Clinic 10/21/2006 120 78 7   Patient has had a persistent cough for 3 weeks
003 HMC         09/01/2006 166 58 8   Patient placed on beta-blockers on 7/1/2006
002 Mayo Clinic 10/01/2006 210 68 9   Patient has been on antibiotics for 10 days
004 HMC         11/11/2006 288 88 9   Patient advised to lose some weight
007 Mayo Clinic 05/01/2006 180 54 7   This patient is always under high stress
050 HMC         07/06/2006 199 60 123 Refer this patient to mental health for evaluation
;


/* Listing of Medical using PROC PRINT
*/

title "Listing of Data Set MEDICAL from PROC PRINT";
proc print data=learn.medical;
     id Patno;
run;

/* Using PROC REPORT (all defaults)
*/

title "Using the REPORT Procedure";
proc report data=learn.medical nowd;
run;

/* Adding a COLUMN statement to PROC REPORT
*/

title "Adding a COLUMN Statement";
proc report data=learn.medical nowd;
      column Patno DX HR Weight;
run;

/* Using PROC REPORT with only numeric variables
*/

title "Report with Only Numeric Variables";
proc report data=learn.medical nowd;
     column HR Weight;
run;

/* Using DEFINE statements to define a display usage
*/

title "Display Usage for Numeric Variables";
proc report data=learn.medical nowd;
      column HR Weight;
      define HR     / display "Heart Rate" width=5;
      define Weight / display              width=6;
run;

/* Specifying a GROUP usage to create a summary report
*/

title "Demonstrating a GROUP Usage";
proc report data=learn.medical nowd;
      column Clinic HR Weight;
      define Clinic / group                              width=11;
      define HR     / analysis mean "Average Heart Rate" width=12  format=5.;
      define Weight / analysis mean "Average Weight"     width=12  format=6.;
run;

/* Demonstrating the FLOW option with PROC REPORT
*/

title "Demonstrating the FLOW Option";
proc report data=learn.medical nowd headline split=' ' ls=74;
      column Patno VisitDate DX HR Weight Comment;
      define Patno     / "Patient Number" width=7;
      define VisitDate / "Visit Date"     width=9 format=date9.;
      define DX        / "DX Code"        width=4 right;
      define HR        / "Heart Rate"     width=6;
      define Weight    /                  width=6;
      define Comment   /                  width=30 flow;
run;

/* Explicitly defining usage for every variable
*/

title "Demonstrating the FLOW Option";
proc report data=learn.medical nowd headline split=' ' ls=74;
      column Patno VisitDate DX HR Weight Comment;
      define Patno     / display "Patient Number" width=7;
      define VisitDate / display "Visit Date"     width=9 format=date9.;
      define DX        / display "DX Code"        width=4 right;
      define HR        / display "Heart Rate"     width=6;
      define Weight    / display                  width=6;
      define Comment   / display                  width=30 flow;
run;

/* Demonstrating the effect of two variables with GROUP usage
*/

title "Multiple GROUP Usages";
proc report data=learn.bicycles nowd headline ls=80;
      column Country Model Units TotalSales;
      define Country    / group                               width=14;
      define Model      / group                               width=13;
      define Units      / sum    "Number of Units"            width=8  format=comma8.;
      define TotalSales / sum    "Total Sales (in thousands)" width=15 format=dollar10.;
run;

/* Reversing the order of variables in the COLUMN statement
*/

title "Multiple GROUP Usages";
proc report data=learn.bicycles nowd headline ls=80;
      column Model Country Manuf Units TotalSales;
      define Country    / group                                width=14;
      define Model      / group                                width=13;
      define Manuf      /                                      width=12;
      define Units      / sum     "Number of Units"            width=8  format=comma8.;
      define TotalSales / sum     "Total Sales (in thousands)" width=15 format=dollar10.;
run;


/* Demonstrating the ORDER usage of PROC REPORT
*/

title "Listing from SALES in EmpID Order";
proc report data=learn.sales nowd headline;
      column EmpID Quantity TotalSales;
      define EmpID      / order "Employee ID" width=11;
      define Quantity   /                     width=8   format=comma8.;
      define TotalSales /       "Total Sales" width=9   format=dollar9.;
run;

/* Applying the ORDER usage for two variables
*/

title "Applying the ORDER Usage for Two Variables";
proc report data=learn.sales nowd headline;
      column EmpID Quantity TotalSales;
      define EmpID      /            order "Employee ID" width=11;
      define TotalSales / descending order "Total Sales" width=9   format=dollar9.;
      define Quantity   /                                width=8   format=comma8.;
run;

/* Creating a multi-column report
*/

title "Random Assignment - Three Groups";
proc report data=learn.sales nowd panels=99 headline ps=16;
      columns EmpID TotalSales;
      define  EmpID        / display width=7;
      define  TotalSales   /         width=5;
run;

/* Requesting a report break (RBREAK statement)
*/

title "Producing Report Breaks";
proc report data=learn.sales nowd headline;
      column Region Quantity TotalSales;
      define Region     /                   width=6;
      define Quantity   / sum               width=8 format=comma8.;
      define TotalSales / sum "Total Sales" width=9 format=dollar9.;
      rbreak after / dol dul summarize;
run;

/* Demonstrating the BREAK statement of PROC REPORT
*/

title "Producing Report Breaks";
proc report data=learn.sales nowd headline;
      column Region Quantity TotalSales;
      define Region      / order             width=6;
      define Quantity    / sum               width=8 format=comma8.;
      define TotalSales  / sum "Total Sales" width=9 format=dollar9.;
      break after region / ol summarize skip;
run;

/* Using a nonprinting variable to order the rows of a report
*/

data temp;
      set learn.sales;
      length LastName $ 10;
      LastName = scan(Name,-1,' ');
run;

title "Listing Ordered by Last Name";
proc report data=temp nowd headline;
      column LastName Name EmpID TotalSales;
      define LastName   / group noprint;
      define Name       / group                      width=15;
      define EmpID      /       "Employee ID" group  width=11;
      define TotalSales / sum   "Total Sales"        width=9 format=dollar9.;
run;

/* Computing a new variable with PROC REPORT
*/

title "Computing a New Variable";
proc report data=learn.medical nowd;
      column Patno Weight WtKg;
      define Patno  / display          "Patient Number" width=7;
      define Weight / display noprint                   width=6;
      define WtKg   / computed         "Weight in Kg"   width=6 format=6.1;
      compute WtKg;
           WtKg = Weight / 2.2;
      endcomp;
run;


/* Computing a Character Variable in a COMPUTE Block
*/

title "Creating a Character Variable in a COMPUTE Block";
proc report data=learn.medical nowd;
      column Patno HR Weight Rate;
      define Patno  / display "Patient Number" width=7;
      define HR     / display "Heart Rate"     width=5;
      define Weight / display                  width=6;
      define Rate   / computed                 width=6;

      compute Rate / character length=6;
           if      HR gt 75        then Rate = 'Fast';
           else if HR gt 55        then Rate = 'Normal';
           else if not missing(HR) then Rate = 'Slow';
      endcomp;
run;

/* Demonstrating an ACROSS usage in PROC REPORT
*/

***Demonstrating an Across Usage;
title "Demonstrating an ACROSS Usage";
proc report data=learn.bicycles nowd headline ls=80;
      column Model,Units Country;
      define Country / group                width=14;
      define Model   / across  "- Model -";
      define Units   / sum     "# of Units" width=14 format=comma8.;
run;

/* Using an ACROSS Usage to Display Statistics
*/

title "Average Blood Counts by Age Group";
proc report data=learn.blood nowd headline;

      column Gender Blood AgeGroup,WBC AgeGroup,RBC;

      define Gender    / group                             width=8;
      define Blood     / group         "Blood Group"       width=8;

      define AgeGroup  / across        "- Age Group -";

      define WBC       / analysis mean                              format=comma8.;
      define RBC       / analysis mean                              format=8.2;
run;

