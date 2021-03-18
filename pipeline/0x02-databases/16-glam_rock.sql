-- lists all bands with Glam as their main style, ranked by their longevity
-- Requirements:
-- 	Column names must be: band_name and lifespan (in years)
-- 	You should use attributes formed and split for computing the lifespan
-- 	Your script can be executed on any database
SELECT band_name , IF(split IS NULL, (2020-formed), (split - formed)) AS lifespan FROM metal_bands
	WHERE style like '%Glam Rock%'
	ORDER BY lifespan DESC;
