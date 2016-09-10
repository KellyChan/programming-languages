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

/* Table Output with csv format
*/
%MACRO freqTable(dataset=,vars=,path=);

ODS CSV FILE = &path;

PROC FREQ DATA = &dataset. ORDER = FREQ;
      TABLES &vars /missing;
RUN;

ODS CSV CLOSE;

%MEND freqTable; 

%freqTable(dataset=nokia.mobilephones,vars=Type,path='G:\eclipseWorkspace\SAS\business\nokia\outputs\holecounts\freq-Type.csv');
%freqTable(dataset=nokia.mobilephones,vars=PhoneModel,path='G:\eclipseWorkspace\SAS\business\nokia\outputs\holecounts\freq-PhoneModel.csv');
%freqTable(dataset=nokia.mobilephones,vars=ScreenType,path='G:\eclipseWorkspace\SAS\business\nokia\outputs\holecounts\freq-ScreenType.csv');
%freqTable(dataset=nokia.mobilephones,vars=Released,path='G:\eclipseWorkspace\SAS\business\nokia\outputs\holecounts\freq-Released.csv');	
%freqTable(dataset=nokia.mobilephones,vars=Quarter,path='G:\eclipseWorkspace\SAS\business\nokia\outputs\holecounts\freq-Quarter.csv');
%freqTable(dataset=nokia.mobilephones,vars=S,path='G:\eclipseWorkspace\SAS\business\nokia\outputs\holecounts\freq-S.csv');
%freqTable(dataset=nokia.mobilephones,vars=Technology,path='G:\eclipseWorkspace\SAS\business\nokia\outputs\holecounts\freq-Technology.csv');	
%freqTable(dataset=nokia.mobilephones,vars=Platform,path='G:\eclipseWorkspace\SAS\business\nokia\outputs\holecounts\freq-Platform.csv');
%freqTable(dataset=nokia.mobilephones,vars=Generation,path='G:\eclipseWorkspace\SAS\business\nokia\outputs\holecounts\freq-Generation.csv');	
%freqTable(dataset=nokia.mobilephones,vars=FormFactor,path='G:\eclipseWorkspace\SAS\business\nokia\outputs\holecounts\freq-FormFactor.csv');	
%freqTable(dataset=nokia.mobilephones,vars=Ringtone,path='G:\eclipseWorkspace\SAS\business\nokia\outputs\holecounts\freq-Ringtone.csv');	
%freqTable(dataset=nokia.mobilephones,vars=Camera,path='G:\eclipseWorkspace\SAS\business\nokia\outputs\holecounts\freq-Camera.csv');	
%freqTable(dataset=nokia.mobilephones,vars=Notes,path='G:\eclipseWorkspace\SAS\business\nokia\outputs\holecounts\freq-Notes.csv');
