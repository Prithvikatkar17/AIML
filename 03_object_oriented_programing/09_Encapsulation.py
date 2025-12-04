# Encapsulation in Python
# Encapsulation is one of the fundamental principles of Object-Oriented Programming (OOP).
# It refers to the bundling of data (attributes) and methods (functions) that operate
# on that data into a single unit known as a class. Encapsulation helps to restrict
# direct access to some of an object's components, which can prevent the accidental 
# modification of data. This is typically achieved using access modifiers.

class BankAccount:
    def __init__(self, account_number, initial_balance=0):
        self.__account_number = account_number  # Private attribute
        self.__balance = initial_balance        # Private attribute

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            print(f"Deposited: ${amount}")
        else:
            print("Deposit amount must be positive.")

    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            print(f"Withdrew: ${amount}")
        else:
            print("Insufficient funds or invalid withdrawal amount.")

    def get_balance(self):
        return self.__balance

    def get_account_number(self):
        return self.__account_number