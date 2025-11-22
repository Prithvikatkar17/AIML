# Write a function to return the sum of digits of a number n.

def sum():
    n = int(input("enter the number:"))
    sum = 0
    while n >0:
        dig=n%10
        n= n//10
        sum +=dig
    print("sum = ",sum)
sum()