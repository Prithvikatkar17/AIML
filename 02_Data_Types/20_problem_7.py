# Write a program that takes a string from the user and prints the number of spaces in the string

str=str(input("enter the string:"))
count=0
for char in str:
    if(char==" "):
        count+=1
print(f"the number of spaces in ginven string are equal to {count}")