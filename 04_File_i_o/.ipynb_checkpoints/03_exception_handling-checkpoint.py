try:
    x=int(input("Please enter a number: "))
    n=10/x

except ZeroDivisionError:
    print("zero not allowed")

except ValueError:
    print("Please enter a valid integer")

else:
    print("The result is:", n)

finally:
    print("Execution completed.")