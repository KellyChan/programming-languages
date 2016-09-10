import pandas as pd

def sorted_table(data, sorted_cols, outfile):

    """Generating a csv table which is sorted by some columns

       inputs:
         - data: it must be pandas.Dataframe()
         - sorted_cols: the table sorted by which columns, i.e. ['Date', 'Name']
         - outfile: the path of the csv file, i.e. "./sorted_table.csv"
    """

    return data.sort_values(by=sorted_cols).to_csv(outfile)
