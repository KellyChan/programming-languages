Scraper: Lowes
=============

### Tasks

Get the catalog from [Lowes](http://www.lowes.com)

- site: www.lowes.com
- info required:
    - category #
    - categories
    - item #
    - model #
    - product #
    - product name
    - product url


### Steps

- step1. analyze the site struture
- step2. parse all urls of department/categories those containing product info
- step3. parse the product info from the urls one by one
- step4. clean the data with standard formats
- step5. load the data into database

### Tools

- language: Python
- libraries: re, beautifulsoup, mechanize

### Notes

- If crawling data is denied by a site, simulating a browser to click the urls.
- Parsing the data from a same container (div).
- URL/href contains a lot of information, especially the data of product ID, dept ID, category ID etc.
- Record all the exceptions, and then re-check.
- It will cost more time if writing the data frequently, but it can record the process well if writing the data by a small batch.


