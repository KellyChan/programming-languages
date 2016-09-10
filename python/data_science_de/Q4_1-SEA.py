"""
Project: Data Science
Subject: Stat - 1. SEA

Author: Kelly Chan
Date: May 7 2014
"""

dataPath = "path/sources/"
tabPath = "path/outputs/stat/tables/"
picPath = "path/outputs/stat/pics/"

import datetime
import pandas as pd
from ggplot import *


def createPeriod(day):
    if day in range(1,11):
        period = 1
    elif day in range(11, 21):
        period = 2
    else:
        period = 3
    return period

def mapPeriod(data):
    labels = {'1': 'First', '2': 'Middle', '3': 'Last'}
    return data['Period'].astype(str).map(labels)

def mapWeekday(data):
    labels = {'0': 'Mon', \
              '1': 'Tue', \
              '2': 'Wed', \
              '3': 'Thu', \
              '4': 'Fri', \
              '5': 'Sat', \
              '6': 'Sun'} 
    return data['Weekday'].astype(str).map(labels)


def loadData(datafile):

    IDs = []
    Sessions = []
    Times = []

    Dates = []
    Periods = []
    WeekDays = []

    Stamps = []
    Hours = []
    Minutes = []
    Seconds = []

    for line in open(datafile, 'rb'):
        thisLine = line.strip().split('\t')
        if (len(thisLine) != 3) or (thisLine[0] == 'No.'):
            continue
        else:
            ID, Session, Time = thisLine

            Date, Stamp = str(Time).split(' ')
            
            Year, Month, Day = (int(x) for x in Date.split('-'))
            Period = createPeriod(Day)
            WeekDay = datetime.date(Year, Month, Day).weekday()

            Hour, Minute, Second = Stamp.split(':')

            IDs.append(ID)
            Sessions.append(Session)
            Times.append(Time)

            Dates.append(Date)
            Periods.append(Period)
            WeekDays.append(WeekDay)

            Stamps.append(Stamp)
            Hours.append(Hour)
            Minutes.append(Minute)
            Seconds.append(Second)

    data = pd.DataFrame({'No.': IDs, \
                         'SessionID': Sessions, \
                         'Time': Times, \
                         'Date': Dates, \
                         'Period': Periods, \
                         'Weekday': WeekDays, \
                         'Stamp': Stamps, \
                         'Hour': Hours, \
                         'Minute': Minutes, \
                         'Second': Seconds})
    data = data.reindex(columns=['No.', 'SessionID', 'Time', \
                                 'Date', 'Period', 'Weekday', \
                                 'Stamp', 'Hour', 'Minute', 'Second'])

    data['Period'] = data['Period'].astype(str)
    data['Weekday'] = data['Weekday'].astype(str)

    return data



def tabSummary(data):
    tab = data.describe()
    tab.to_csv(tabPath + 'summary.csv')
    return tab
    

def tabFreq(data, col):
    tab = data[col].value_counts().reset_index()
    tab.columns = ['value', 'freq']
    
    tab['percent'] = tab['freq'] / sum(tab['freq'])

    tab.to_csv("%sholecount-%s.csv" % (tabPath, col))
    
    return tab

def tabCross(rows, cols, tabName):
    tab = pd.crosstab(rows=rows, cols=cols, margins=True)
    tab.to_csv("%scrosstab-%s.csv" % (tabPath, tabName))
    return tab    


def plotHourlyVisitsWeekday():

    data = pd.read_csv(tabPath + "crosstab-HourWeekday.csv")
    data = data.iloc[0:24, 0:8]

    data_melt = pd.melt(data[['Hour', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']], id_vars='Hour')
    data_melt.columns = ['Hour', 'Weekday', 'Frequency']
    p = ggplot(aes(x='Hour', y='Frequency', colour='Weekday'), data=data_melt) 
    p = p + geom_line(size=2)
    p = p + ggtitle("Hourly visits by weekday")
    p = p + xlab("Hour")
    p = p + ylab("Frequency")
    ggsave(p, "%sHourly_visits_by_weekday.png" % picPath)

def plotAll():

    plotHourlyVisitsWeekday()



def main():

    #data = loadData(dataPath + 'webtrekk_report_2012-11-01_Visit_IDs.csv')
    #data = data[data['Date'] != '2012-10-31']

    #data['Period'] = mapPeriod(data)
    #data['Weekday'] = mapWeekday(data)

    #data.to_csv(tabPath + 'clean_data.csv')

    #tabSummary(data)

    #for col in data.columns:
    #    tabFreq(data, col)

    #tabCross([data['Date']], [data['Hour']], 'DateHour')
    #tabCross([data['Hour']], [data['Period']], 'HourPeriod')
    #tabCross([data['Hour']], [data['Weekday']], 'HourWeekday')

    plotAll()



if __name__ == '__main__':
    main()
