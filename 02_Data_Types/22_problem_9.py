# Given a list, print all elements that appear more than once in the list.
#Hint:use sets
list = [1, 2, 3, 4, 2, 5, 1, 6, 3]
seen = set()
duplicates = set()
for item in list:
    if item in seen:
        duplicates.add(item)
    else:
        seen.add(item)

print("Elements that appear more than once:", duplicates)