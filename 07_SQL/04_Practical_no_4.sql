

CREATE TABLE Sales (
    SaleID INT PRIMARY KEY,
    ProductName VARCHAR(50),
    Category VARCHAR(30),
    Quantity INT,
    Price INT,
    City VARCHAR(30)
);

INSERT INTO Sales VALUES (1, 'Laptop', 'Electronics', 2, 50000, 'Pune');
INSERT INTO Sales VALUES (2, 'Mobile', 'Electronics', 5, 20000, 'Mumbai');
INSERT INTO Sales VALUES (3, 'Table', 'Furniture', 3, 7000, 'Delhi');
INSERT INTO Sales VALUES (4, 'Chair', 'Furniture', 6, 3000, 'Pune');
INSERT INTO Sales VALUES (5, 'TV', 'Electronics', 4, 40000, 'Mumbai');
INSERT INTO Sales VALUES (6, 'Sofa', 'Furniture', 2, 25000, 'Chennai');
INSERT INTO Sales VALUES (7, 'Headphones', 'Electronics', 8, 2000, 'Delhi');


SELECT COUNT(*) AS Total_Sales FROM Sales;


SELECT SUM(Quantity) AS Total_Quantity FROM Sales;


SELECT AVG(Price) AS Average_Price FROM Sales;


SELECT MAX(Price) AS Highest_Price FROM Sales;


SELECT MIN(Price) AS Lowest_Price FROM Sales;

SELECT Category, SUM(Quantity) AS Total_Quantity
FROM Sales
GROUP BY Category;


SELECT Category, AVG(Price) AS Avg_Price
FROM Sales
GROUP BY Category;


SELECT Category, SUM(Quantity * Price) AS Total_Sales_Value
FROM Sales
GROUP BY Category;


SELECT City, COUNT(*) AS Sales_Count
FROM Sales
GROUP BY City;






SELECT Category, SUM(Quantity) AS Total_Quantity
FROM Sales
GROUP BY Category
HAVING SUM(Quantity) > 10;


SELECT Category, AVG(Price) AS Avg_Price
FROM Sales
GROUP BY Category
HAVING AVG(Price) > 20000;


SELECT Category, SUM(Quantity * Price) AS Total_Sales_Value
FROM Sales
WHERE Price > 5000
GROUP BY Category
HAVING SUM(Quantity * Price) > 100000;
