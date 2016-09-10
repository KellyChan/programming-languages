/*
Query the list of CITY names starting with vowels (a, e, i, o, u) from STATION. Your result cannot contain duplicates.
*/

SELECT DISTINCT(CITY)
FROM STATION
WHERE CITY REGEXP '^[AEIOU].'
ORDER BY CITY


/*
Query the list of CITY names ending with vowels (a, e, i, o, u) from STATION. Your result cannot contain duplicates.
*/
SELECT DISTINCT(CITY)
FROM STATION
WHERE CITY REGEXP '.[aeiou]$'
ORDER BY CITY


/*
Query the list of CITY names from STATION which have vowels as both their first and last characters. Your result cannot contain duplicates.
*/
SELECT DISTINCT(CITY)
FROM STATION
WHERE CITY REGEXP '^[aeiou].*[aeiou]$'
ORDER BY CITY

/*
Query the list of CITY names from STATION that do not start with vowels. Your result cannot contain duplicates.
*/
SELECT DISTINCT(CITY)
FROM STATION
WHERE CITY NOT REGEXP '^[aeiou].*'
ORDER BY CITY


/*
Query the list of CITY names from STATION that do not end with vowels. Your result cannot contain duplicates.
*/
SELECT DISTINCT(CITY)
FROM STATION
WHERE CITY NOT REGEXP '.*[aeiou]$'
ORDER BY CITY



/*
Query the list of CITY names from STATION that either do not start with vowels or do not end with vowels. Your result cannot contain duplicates.
*/
SELECT DISTINCT(CITY)
FROM STATION
WHERE CITY NOT REGEXP '^[aeiou].*[aeiou]$'
ORDER BY CITY


/*
Query the list of CITY names from STATION that do not start with vowels and do not end with vowels. Your result cannot contain duplicates.
*/
SELECT DISTINCT(CITY)
FROM STATION
WHERE CITY REGEXP '^[^aeiou].*[^aeiou]$'
ORDER BY CITY
