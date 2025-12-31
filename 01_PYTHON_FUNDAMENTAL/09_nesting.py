
username = input("Enter your username: ")
passward = input("Enter your passward: ")


if(username=="admin"):
    if(passward=="1234"):
        print("Welcome Admin")
    elif(passward!="1234"):
        print("Invalid Passward")
else:
    print("Invalid Username")
# This program demonstrates nested IF statements
# to validate username and password.     