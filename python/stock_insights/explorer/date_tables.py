import pandas as pd

def first_date_table(data, date_col, subset_col, outfile):

    """Output a csv table by keeping the records of the first date only

       inputs:
         - data: it must be pandas.Dataframe()
         - date_col: the column of the Date, i.e. ['Date']
         - subset_col: the column of the subset/ID, i.e. 'Name'
         - outfile: the path of the csv table, i.e. "./first_dates.csv"
    """

    first_dates = data.sort_values(by=date_col).drop_duplicates(subset=subset_col, keep='first')
    first_dates.to_csv(outfile)


def last_date_table(data, date_col, subset_col, outfile):

    """Output a csv table by keeping the records of the first date only

       inputs:
         - data: it must be pandas.Dataframe()
         - date_col: the column of the Date, i.e. ['Date']
         - subset_col: the column of the subset/ID, i.e. 'Name'
         - outfile: the path of the csv table, i.e. "./last_dates.csv"
    """

    last_dates = data.sort_values(by=date_col).drop_duplicates(subset=subset_col, keep='last')
    last_dates.to_csv(outfile)
