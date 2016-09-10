import pandas as pd

def gen_freq_table(data, col, outfile):

    """generate the frequency table by a column and return a csv table
       
       inputs:
           - data: convert to panda.Dataframe()
           - col: a string, i.e. "Date"
           - outfile: the path of the csv table, i.e. "./freq.csv"
    """

    return data[col].value_counts().to_csv(outfile)
