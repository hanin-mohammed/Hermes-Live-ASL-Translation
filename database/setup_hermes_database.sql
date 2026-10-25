CREATE DATABASE IF NOT EXISTS hermes_db;

USE hermes_db;

CREATE TABLE IF NOT EXISTS signs (
    sign_name VARCHAR(20) PRIMARY KEY,
    sign_information VARCHAR(255),
    speech_text VARCHAR(255)
);

SET @speech_column_exists = (
    SELECT COUNT(*)
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'hermes_db'
      AND TABLE_NAME = 'signs'
      AND COLUMN_NAME = 'speech_text'
);

SET @add_speech_column = IF(
    @speech_column_exists = 0,
    'ALTER TABLE signs ADD COLUMN speech_text VARCHAR(255)',
    'SELECT 1'
);

PREPARE add_speech_column_statement FROM @add_speech_column;
EXECUTE add_speech_column_statement;
DEALLOCATE PREPARE add_speech_column_statement;

INSERT INTO signs (sign_name, sign_information, speech_text)
VALUES
    ('A', 'This sign represents the letter A.', 'a'),
    ('B', 'This sign represents the letter B.', 'b'),
    ('C', 'This sign represents the letter C.', 'c'),
    ('1', 'This sign represents the number 1.', 'one'),
    ('2', 'This sign represents the number 2.', 'two'),
    ('3', 'This sign represents the number 3.', 'three'),
    ('ILY', 'This sign means I love you.', 'I love you')
ON DUPLICATE KEY UPDATE
    sign_information = VALUES(sign_information),
    speech_text = VALUES(speech_text);

DELETE FROM signs
WHERE sign_name NOT IN ('A', 'B', 'C', '1', '2', '3', 'ILY');
