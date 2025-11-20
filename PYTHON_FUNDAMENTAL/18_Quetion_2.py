salary = int(input("Enter the salary:"))

if salary <= 30000:
    print(f"tax =", salary*0.05)
elif salary <70000:
    print(f"tax=",salary*0.15)
else:
    print(f"tax=",salary*0.25)