/*
Query the two cities in STATION with the shortest and longest CITY names, as well as their respective lengths (i.e.: number of characters in the name). If there is more than one smallest or largest city, choose the one that comes first when ordered alphabetically.
*/

SELECT *
FROM (
         SELECT CITY, LENGTH(CITY) AS LEN
         FROM STATION
         ORDER BY LEN ASC, CITY ASC
         LIMIT 1
     ) AS MINCITY

UNION ALL

SELECT *
FROM (
        SELECT CITY, LENGTH(CITY) AS LEN
        FROM STATION
        ORDER BY LEN DESC, CITY ASC
        LIMIT 1
     ) AS MAXCITY
