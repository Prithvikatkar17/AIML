# Write a program that takes  as input. Using conditional statements, 
# calculate the  final tax rate based on these rules:
#  • If salary < 30,000 → 5%
#  • If salary is 30,000–70,000 → 15%
#  • If salary > 70,000 → 25%

salary = int(input("Enter the salary:"))

if salary <= 30000:
    print(f"tax =", salary*0.05)
elif salary <70000:
    print(f"tax=",salary*0.15)
else:
    print(f"tax=",salary*0.25)