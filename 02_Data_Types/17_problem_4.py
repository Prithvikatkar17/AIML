# Given a tuple of integers, create:
# • A tuple of all even numbers
# • A tuple of all odd numbers

tup=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
even_tup=()
odd_tup=()
for i in tup:
    if i%2==0:
        even_tup += (i,)
    else:
        odd_tup += (i,)
print(f"Even numbers tuple: {even_tup}")
print(f"Odd numbers tuple: {odd_tup}")