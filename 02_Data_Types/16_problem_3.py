# Input two lists of integers from the user. Merge them into one list and sort the resulted list.
list1 = []
list2 = []

n1 = int(input("Enter the number of elements in the first list: "))
for i in range(n1):
    elem = int(input(f"Enter element {i+1} of the first list: "))
    list1.append(elem)

n2 = int(input("Enter the number of elements in the second list: "))
for i in range(n2):
    elem = int(input(f"Enter element {i+1} of the second list: "))
    list2.append(elem)
    
merged_list = list1 + list2
merged_list.sort()
print("Merged and sorted list:", merged_list)
