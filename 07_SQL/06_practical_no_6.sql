CREATE DATABASE IF NOT EXISTS College;
USE College;

CREATE TABLE Student (
    Roll_no    INT PRIMARY KEY,
    Name       VARCHAR(50),
    Department VARCHAR(50),
    Year       INT
);

CREATE TABLE Subject (
    Sub_code   INT PRIMARY KEY,
    Sub_name   VARCHAR(50),
    Department VARCHAR(50)
);

CREATE TABLE Marks (
    Roll_no  INT,
    Sub_code INT,
    Marks    INT,
    FOREIGN KEY (Roll_no)  REFERENCES Student(Roll_no),
    FOREIGN KEY (Sub_code) REFERENCES Subject(Sub_code)
);

INSERT INTO Student VALUES (1, 'Amit',    'Computer', 2);
INSERT INTO Student VALUES (2, 'Raj',     'Computer', 3);
INSERT INTO Student VALUES (3, 'Priya',   'IT',       2);
INSERT INTO Student VALUES (4, 'Sneha',   'IT',       3);
INSERT INTO Student VALUES (5, 'Karan',   'Mechanical', 2);

INSERT INTO Subject VALUES (101, 'DBMS',       'Computer');
INSERT INTO Subject VALUES (102, 'OS',         'Computer');
INSERT INTO Subject VALUES (103, 'Networks',   'IT');
INSERT INTO Subject VALUES (104, 'Math',       'Mechanical');

INSERT INTO Marks VALUES (1, 101, 85);
INSERT INTO Marks VALUES (1, 102, 78);
INSERT INTO Marks VALUES (2, 101, 92);
INSERT INTO Marks VALUES (2, 102, 88);
INSERT INTO Marks VALUES (3, 101, 75);
INSERT INTO Marks VALUES (3, 103, 85);
INSERT INTO Marks VALUES (4, 103, 90);
INSERT INTO Marks VALUES (4, 102, 60);
INSERT INTO Marks VALUES (5, 104, 55);

SELECT S.Name, M.Marks
FROM Student S
JOIN Marks M ON S.Roll_no = M.Roll_no
WHERE M.Marks = (SELECT MAX(Marks) FROM Marks);

SELECT S.Name, M.Marks
FROM Student S
JOIN Marks M ON S.Roll_no = M.Roll_no
WHERE M.Marks > (SELECT AVG(Marks) FROM Marks);

SELECT S.*, M.Marks
FROM Student S
JOIN Marks M ON S.Roll_no = M.Roll_no
WHERE M.Marks = (SELECT MIN(Marks) FROM Marks);

SELECT S.Name, M.Marks
FROM Student S
JOIN Marks M ON S.Roll_no = M.Roll_no
WHERE M.Marks IN (
    SELECT M2.Marks
    FROM Marks M2
    JOIN Student S2 ON M2.Roll_no = S2.Roll_no
    WHERE S2.Name = 'Amit'
);

SELECT DISTINCT S.Name, S.Department
FROM Student S
JOIN Marks M ON S.Roll_no = M.Roll_no
JOIN Subject Sub ON M.Sub_code = Sub.Sub_code
WHERE Sub.Department = 'Computer';

SELECT DISTINCT S.Name, M.Marks
FROM Student S
JOIN Marks M ON S.Roll_no = M.Roll_no
WHERE M.Marks > ANY (
    SELECT M2.Marks
    FROM Marks M2
    JOIN Subject Sub ON M2.Sub_code = Sub.Sub_code
    WHERE Sub.Sub_name = 'DBMS'
);

SELECT S.Name, Sub.Sub_name, M.Marks
FROM Student S
JOIN Marks M ON S.Roll_no = M.Roll_no
JOIN Subject Sub ON M.Sub_code = Sub.Sub_code
WHERE M.Marks > (
    SELECT AVG(M2.Marks)
    FROM Marks M2
    WHERE M2.Sub_code = M.Sub_code
);

SELECT Sub.Sub_name, MAX(M.Marks) AS Highest_Marks
FROM Marks M
JOIN Subject Sub ON M.Sub_code = Sub.Sub_code
GROUP BY Sub.Sub_code, Sub.Sub_name;