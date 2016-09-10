Crawler: Yelp
============

### Functions

The functions of this project are separated by two parts:
- crawling data by bizlist
- crawling data by search pages

Processing steps:
- 1. crawling the data from the web
- 2. cleansing the data
- 3. output the cleaned data


### Folder Structure

    .
    |
    |--- [docs]
    |       |----- RegExEval-DataEngineer.pdf
    |
    |--- [data]
    |       |----- yelp_bizlist_raw.txt
    |       |----- yelp_bizlist_cleaned.tsv
    |       |----- yelp_searchpages_raw.txt
    |       |----- yelp_searchpages_cleaned.tsv
    |
    |--- [src]
            |----- crawler_yelp_bizlist.py
            |----- cleaning_yelp_bizlist.py
            |----- crawler_yelp_searchpages.py
            |----- cleaning_yelp_searchpages.py
