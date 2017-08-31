# IS362


#Question #1
SELECT
COUNT(speed) AS 'Number Listed Speed',
MAX(speed) AS 'Max Speed',
MIN(speed) AS 'Min Speed'
FROM planes;

#2
SELECT SUM(distance) AS 'Total Distance'
FROM flights
WHERE year = 2013
AND month = 1;

#2 Part 2
SELECT SUM(distance) AS 'Total Distance Null'
FROM flights
WHERE year = 2013
AND month = 1
AND tailnum is null;

#3
SELECT
SUM(distance) AS 'Total Distance',
planes.manufacturer
FROM flights 
INNER  JOIN planes 
ON flights.tailnum = planes.tailnum
WHERE (flights.month = 7) AND (flights.day = 5) AND (flights.year = 2013)
GROUP BY planes.manufacturer;

#3 Part 2
SELECT
SUM(distance) AS 'Total Distance',
planes.manufacturer
FROM flights 
LEFT OUTER JOIN planes 
ON flights.tailnum = planes.tailnum
WHERE (flights.month = 7) AND (flights.day = 5) AND (flights.year = 2013)
GROUP BY planes.manufacturer;


#4 How many flights to LAX  by airline using what model plane and manufacturer in Jan. 2013?
SELECT 
COUNT(flights.tailnum) as 'Number of Flights', airlines.name as 'Airline', planes.model, planes.manufacturer, dest
FROM flights
LEFT JOIN planes ON flights.tailnum = planes.tailnum
LEFT JOIN airlines  ON flights.carrier = airlines.carrier
WHERE (flights.year = 2013) AND (flights.month = 1) and (flights.dest = 'LAX')
GROUP BY airlines.name;


#Part 2
SELECT year, month, day, dep_delay, carrier, tailnum, dest
FROM flights
GROUP by carrier
INTO OUTFILE '/Users/Nicholas/Desktop/flightsdelay.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';
