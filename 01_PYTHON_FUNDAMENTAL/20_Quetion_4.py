# Write a function that prints the  digits of a number, n.

n = int(input("enter tne number:"))

while n > 0:
    lastdig = n%10
    n = n//10
    print(" " ,lastdig)

m=int(input("enter the number :"))
while m > 0:
    s = str(m)                # convert to string
    dig = m // 10**(len(s)-1) # extract first digit
    m= m - dig * (10**(len(s)-1))  # remove first digit
    print(" ", dig)

