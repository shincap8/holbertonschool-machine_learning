-- Import in hbtn_0c_0 database this table dump
-- Write a script that displays the max temperature of each state (ordered by State name).
USE hbtn_0c_0;
SELECT state, MAX(value) AS max_temp FROM temperatures GROUP BY state ORDER BY state ASC LIMIT 3;
