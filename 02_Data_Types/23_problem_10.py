# Ask the user for a string and print:
# •All unique characters
# •The count of unique characters
user_input = input("Enter a string: ")
unique_chars = set(user_input)
print("Unique characters:", unique_chars)
print("Count of unique characters:", len(unique_chars))
