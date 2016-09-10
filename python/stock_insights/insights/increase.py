import pandas as pd

def cal_increase(data, outfile):

    """Calculate the increase value and return the calculated table for all stocks
    
       formula:
          increase = (the last value - the first value) / the first value * 100

       conditions:
          - data columns: ['Name', 'Date', 'notes', 'Value', 'Change']
          - data must be cleaned (without any missing values such as Null, UNKNOWN, NA)
          - data['Date'] must be converted to be the type 'datetime' by `pd.to_datetime()`
    """

    # the records of the first date
    first_dates = data.sort_values(by=['Date']).drop_duplicates(subset='Name', keep='first')
    # the records of the last date
    last_dates = data.sort_values(by=['Date']).drop_duplicates(subset='Name', keep='last')
    # merge the records by 'Stock Name' with first_date_records and last_date_records
    final_data = pd.merge(first_dates, last_dates, on='Name')

    # calculate the increased values
    # increase = (the value of the last date - the value of the first date) / the value of the first date * 100
    final_data['increased'] = (final_data['Value_y'].astype(float) - final_data['Value_x'].astype(float)) / final_data['Value_x'].astype(float) * 100

    # rename the columns
    final_data.rename(columns={'Date_x': 'Date_first', \
                               'notes_x': 'notes_first', \
                               'Value_x': 'Value_first', \
                               'Change_x': 'Change_first', \
                               'Date_y': 'Date_last', \
                               'notes_y': 'notes_last', \
                               'Value_y': 'Value_last', \
                               'Change_y': 'Change_last'}, inplace=True)

    # output the results
    final_data.sort_values(by=['increased'], ascending=False).to_csv(out_file)
