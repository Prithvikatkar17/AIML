use  college;

CREATE TABLE Employee (
    EmpID INT PRIMARY KEY,
    EmpName VARCHAR(50),
    Salary INT,
    Department VARCHAR(50)
);

INSERT INTO Employee VALUES
(1, 'Amit', 25000, 'HR'),
(2, 'Neha', 30000, 'IT'),
(3, 'Raj', 20000, 'HR'),
(4, 'Simran', 40000, 'IT'),
(5, 'Karan', 15000, 'Sales');

DELIMITER //

CREATE PROCEDURE IncreaseSalary()
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE emp_id INT;
    DECLARE emp_salary INT;
    DECLARE emp_cursor CURSOR FOR SELECT EmpID, Salary FROM Employee;
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;
    OPEN emp_cursor;
    read_loop: LOOP
        FETCH emp_cursor INTO emp_id, emp_salary;
        IF done = 1 THEN LEAVE read_loop; END IF;
        UPDATE Employee SET Salary = emp_salary + (emp_salary * 0.10) WHERE EmpID = emp_id;
    END LOOP;
    CLOSE emp_cursor;
END //

CREATE PROCEDURE DeptCount()
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE dept_name VARCHAR(50);
    DECLARE dept_cursor CURSOR FOR SELECT DISTINCT Department FROM Employee;
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;
    OPEN dept_cursor;
    read_loop: LOOP
        FETCH dept_cursor INTO dept_name;
        IF done = 1 THEN LEAVE read_loop; END IF;
        SELECT dept_name AS Department, COUNT(*) AS TotalEmployees FROM Employee WHERE Department = dept_name;
    END LOOP;
    CLOSE dept_cursor;
END //

CREATE FUNCTION TotalSalary()
RETURNS INT
DETERMINISTIC
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE total INT DEFAULT 0;
    DECLARE sal INT;
    DECLARE sal_cursor CURSOR FOR SELECT Salary FROM Employee;
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;
    OPEN sal_cursor;
    read_loop: LOOP
        FETCH sal_cursor INTO sal;
        IF done = 1 THEN LEAVE read_loop; END IF;
        SET total = total + sal;
    END LOOP;
    CLOSE sal_cursor;
    RETURN total;
END //

CREATE PROCEDURE IncreaseSalaryIT()
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE emp_id INT;
    DECLARE emp_salary INT;
    DECLARE emp_dept VARCHAR(50);
    DECLARE emp_cursor CURSOR FOR SELECT EmpID, Salary, Department FROM Employee;
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;
    OPEN emp_cursor;
    read_loop: LOOP
        FETCH emp_cursor INTO emp_id, emp_salary, emp_dept;
        IF done = 1 THEN LEAVE read_loop; END IF;
        IF emp_dept = 'IT' THEN
            UPDATE Employee SET Salary = emp_salary + (emp_salary * 0.20) WHERE EmpID = emp_id;
        END IF;
    END LOOP;
    CLOSE emp_cursor;
END //

CREATE PROCEDURE FindHighestSalary()
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE emp_id INT;
    DECLARE emp_name VARCHAR(50);
    DECLARE emp_salary INT;
    DECLARE max_salary INT DEFAULT 0;
    DECLARE max_emp_id INT;
    DECLARE max_name VARCHAR(50);
    DECLARE emp_cursor CURSOR FOR SELECT EmpID, EmpName, Salary FROM Employee;
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;
    OPEN emp_cursor;
    read_loop: LOOP
        FETCH emp_cursor INTO emp_id, emp_name, emp_salary;
        IF done = 1 THEN LEAVE read_loop; END IF;
        IF emp_salary > max_salary THEN
            SET max_salary = emp_salary;
            SET max_emp_id = emp_id;
            SET max_name = emp_name;
        END IF;
    END LOOP;
    CLOSE emp_cursor;
    SELECT max_emp_id AS EmpID, max_name AS EmpName, max_salary AS HighestSalary;
END //

DELIMITER ;

CALL IncreaseSalary();
SELECT * FROM Employee;

CALL DeptCount();

SELECT TotalSalary();

CALL IncreaseSalaryIT();
SELECT * FROM Employee;

CALL FindHighestSalary();