-- creates a stored procedure AddBonus that adds a new correction for a student.
-- Requirements:
--	Procedure AddBonus is taking 3 inputs (in this order):
--	user_id, a users.id value (you can assume user_id is linked to an existing users)
--	project_name, a new or already exists projects - if no projects.name found in the table, you should create it
--	score_new, the score value for the correction
delimiter //
CREATE PROCEDURE AddBonus(
	IN user_id_new INTEGER,
	IN project_name varchar(255), 
	IN score_new INTEGER)
	BEGIN
		IF NOT EXISTS (SELECT name FROM projects WHERE name=project_name) THEN
			INSERT INTO projects(name) VALUES (project_name);
		END IF;
		INSERT INTO corrections(user_id, project_id, score)
			VALUES(user_id_new,
			      (SELECT id FROM projects WHERE name = project_name),
			      score_new);
	END //
delimiter;
