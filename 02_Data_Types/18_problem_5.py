# Create a dictionary where:
# •Keys=student names
# •Values=marks(integer)
# Write a menu-based program where user presses a key (ʼAʼ, ‘Bʼ, ‘Cʼ, ‘Dʼ) 
# depending on the operation they want to perform on the dictionary:
# 1.A-Add a student
# 2.B-Update marks
# 3.C-Search for a student
# 4.D-Display all students and marks

students = {}
while True:
    print("\nMenu:")
    print("A - Add a student")
    print("B - Update marks")
    print("C - Search for a student")
    print("D - Display all students and marks")
    print("E - Exit")
    
    choice = input("Enter your choice: ").upper()
    
    if choice == 'A':
        name = input("Enter student name: ")
        marks = int(input("Enter marks: "))
        students[name] = marks
        print(f"Student {name} added with marks {marks}.")
        
    elif choice == 'B':
        name = input("Enter student name to update marks: ")
        if name in students:
            marks = int(input("Enter new marks: "))
            students[name] = marks
            print(f"Marks for {name} updated to {marks}.")
        else:
            print(f"Student {name} not found.")
            
    elif choice == 'C':
        name = input("Enter student name to search: ")
        if name in students:
            print(f"{name} has marks: {students[name]}")
        else:
            print(f"Student {name} not found.")
            
    elif choice == 'D':
        if students:
            print("Students and their marks:")
            for name, marks in students.items():
                print(f"{name}: {marks}")
        else:
            print("No students in the record.")
            
    elif choice == 'E':
        print("Exiting the program.")
        break
        
    else:
        print("Invalid choice. Please try again.")
