

use  college;
CREATE TABLE Customer (
    CustomerID INT PRIMARY KEY,
    CustomerName VARCHAR(50),
    City VARCHAR(30)
);

CREATE TABLE Orders (
    OrderID INT PRIMARY KEY,
    CustomerID INT,
    Product VARCHAR(50),
    Amount INT,
    FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID)
);


INSERT INTO Customer VALUES (1, 'Rohit', 'Pune');
INSERT INTO Customer VALUES (2, 'Anjali', 'Mumbai');
INSERT INTO Customer VALUES (3, 'Suresh', 'Delhi');
INSERT INTO Customer VALUES (4, 'Meena', 'Chennai');

INSERT INTO Orders VALUES (101, 1, 'Laptop', 60000);
INSERT INTO Orders VALUES (102, 1, 'Mouse', 1500);
INSERT INTO Orders VALUES (103, 2, 'Mobile', 25000);
INSERT INTO Orders VALUES (104, 3, 'Table', 8000);



SELECT Customer.CustomerName, Orders.Product, Orders.Amount
FROM Customer
INNER JOIN Orders
ON Customer.CustomerID = Orders.CustomerID;


SELECT Customer.CustomerName, Orders.Product, Orders.Amount
FROM Customer
LEFT JOIN Orders
ON Customer.CustomerID = Orders.CustomerID;


SELECT Customer.CustomerName, Orders.Product, Orders.Amount
FROM Customer
RIGHT JOIN Orders
ON Customer.CustomerID = Orders.CustomerID;


SELECT Customer.CustomerName, Orders.Product, Orders.Amount
FROM Customer
LEFT JOIN Orders
ON Customer.CustomerID = Orders.CustomerID
UNION
SELECT Customer.CustomerName, Orders.Product, Orders.Amount
FROM Customer
RIGHT JOIN Orders
ON Customer.CustomerID = Orders.CustomerID;


SELECT Customer.CustomerName, Orders.Product
FROM Customer
CROSS JOIN Orders;


CREATE VIEW Customer_Order_View AS
SELECT Customer.CustomerName, Customer.City, Orders.Product, Orders.Amount
FROM Customer
INNER JOIN Orders
ON Customer.CustomerID = Orders.CustomerID;


SELECT * FROM Customer_Order_View;


UPDATE Customer_Order_View
SET Amount = 65000
WHERE Product = 'Laptop';

DROP VIEW Customer_Order_View;

SET SQL_SAFE_UPDATES = 1;