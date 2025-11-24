# Write a program to check whether two lists share no common elements or not.
# hint :use sets.

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
set1 = set(list1)
set2 = set(list2)
n1=len(set1)
n2=len(set2)
sum1=n1+n2
set3=set1.union(set2)
n3=len(set3)
if sum1==n3:
    print("The two lists share no common elements.")
else:
    print("The two lists have common elements.")