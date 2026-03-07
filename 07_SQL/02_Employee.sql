CREATE TABLE Employee (
    EmpID INT PRIMARY KEY,
    Name VARCHAR(50),
    Department VARCHAR(30),
    Salary INT,
    City VARCHAR(30)
);
INSERT INTO Employee VALUES (101, 'Amit', 'IT', 40000, 'Pune');
INSERT INTO Employee VALUES (102, 'Neha', 'HR', 35000, 'Mumbai');
INSERT INTO Employee VALUES (103, 'Rahul', 'IT', 45000, 'Delhi');

select * from employee;

UPDATE Employee
SET Salary = Salary + 5000
WHERE Department = 'IT';


SET SQL_SAFE_UPDATES = 1;

DELETE FROM Employee
WHERE City = 'Delhi';

SELECT * FROM Employee
WHERE Department = 'IT' AND Salary > 40000;

SELECT Name FROM Employee WHERE Department = 'IT'
UNION
SELECT Name FROM Employee WHERE Department = 'HR';

CREATE USER user1 IDENTIFIED BY 'pass123';
GRANT SELECT, INSERT ON Employee TO user1;
REVOKE INSERT ON Employee FROM user1;

CREATE ROLE manager_role;
GRANT SELECT, UPDATE ON Employee TO manager_role;

GRANT manager_role TO user1;


START TRANSACTION;

INSERT INTO Employee VALUES (104, 'Priya', 'Finance', 38000, 'Chennai');

SAVEPOINT sp1;

UPDATE Employee
SET Salary = 50000
WHERE EmpID = 104;

ROLLBACK TO sp1;

COMMIT;
