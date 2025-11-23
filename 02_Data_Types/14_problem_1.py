# Ask the user for a string and check whether it is a palindrome or not.
user_input = input("Enter a string: ")
reversed_input = ""
for char in reversed(user_input):
    reversed_input += char
    
if user_input == reversed_input:
    print(f'"{user_input}" is a palindrome.')
else:
    print(f'"{user_input}" is not a palindrome.')

