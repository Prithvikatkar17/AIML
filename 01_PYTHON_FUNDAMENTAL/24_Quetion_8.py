# Design a program to continuously input a number  from user & print if it is 
# positive or negative until the user enters “Quit”.

def PosNeg():
    while True:
        n = input("Enter a number (or type 'quit' to exit): ")

        if n.lower() == "quit":
            print("Thank you!")
            break

        # convert to integer AFTER checking quit
        n = int(n)

        if n > 0:
            print("Positive")
        elif n < 0:
            print("Negative")
        else:
            print("Zero")


PosNeg()

    