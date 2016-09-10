/*

PROGRAM NAME: Nokia Anaysis - Mobile Phones

PROGRAMMER: Kelly Chan
Date Written: Aug 23 2013

*/

libname nokia 'G:\eclipseWorkspace\SAS\business\nokia\libs';
libname formats 'G:\eclipseWorkspace\SAS\business\nokia\libs';
options fmtsearch=(formats);

/* Data Imported
*/

PROC IMPORT DATAFILE = "G:\eclipseWorkspace\SAS\business\nokia\data\mobilephones.csv"
              OUT     = nokia.mobilephones
              DBMS    = csv
              REPLACE;
      GETNAMES = YES;
RUN;

/* Table Output with HTML format
*/

ods html body = 'body_holecounts.html'
         contents = 'contents_holecounts.html'
         frame = 'frame_holecounts.html'
         path = 'G:\eclipseWorkspace\SAS\business\nokia\outputs\holecounts' (url=none);

/* Holecount Tables by Variables
*/

title "Nokia Mobile Phones - frequency counts";

proc freq data=nokia.mobilephones;
         tables  Type         
                 PhoneModel   
                 ScreenType	  
                 Released	  
                 Quarter	  
                 S	          
                 Technology	  
                 Platform	  
                 Generation	  
                 FormFactor	  
                 Ringtone	  
                 Camera	      
                 Notes       /missing
                 ;
run; 

ods html close;
