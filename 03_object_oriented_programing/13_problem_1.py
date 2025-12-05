# Create a BankAccount class with account_number,
#  owner_name, and balance. Add deposit, withdraw, and check_balance methods

class BankAccount:
    owner_name = "John Doe"  # Class attribute
    account_number = "123456789"  # Class attribute
    balance = 0.0  # Class attribute

    def deposit(amount):
        if amount > 0:
            BankAccount.balance += amount
            print(f"Deposited: ${amount}")
        else:
            print("Deposit amount must be positive.")
    
    def withdraw(amount):
        if BankAccount.balance > amount:
            BankAccount.balance -= amount
            print(f"Withdrew: ${amount}")

    def check_balance():
        return BankAccount.balance
    
# Example usage:
BankAccount.deposit(500)    
print(f"Current Balance: ${BankAccount.check_balance()}")
BankAccount.withdraw(200)
print(f"Current Balance: ${BankAccount.check_balance()}")