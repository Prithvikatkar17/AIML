# Create a student class with private attributes _name, _roll_no, and _marks. 
# Provide getter and setter methods with validation 
# (e.g., marks cannot be negative, roll number has to be between 1 & 100 & name cannot be empty)

class Student:
    def __init__(self, name, roll_no, marks):
        self.set_name(name)
        self.set_roll_no(roll_no)
        self.set_marks(marks)

    def get_name(self):
        return self._name

    def set_name(self, name):
        if name:
            self._name = name
        else:
            raise ValueError("Name cannot be empty.")

    def get_roll_no(self):
        return self._roll_no

    def set_roll_no(self, roll_no):
        if 1 <= roll_no <= 100:
            self._roll_no = roll_no
        else:
            raise ValueError("Roll number must be between 1 and 100.")

    def get_marks(self):
        return self._marks

    def set_marks(self, marks):
        if marks >= 0:
            self._marks = marks
        else:
            raise ValueError("Marks cannot be negative.")
        
# Example usage:
student = Student("John Doe", 25, 85)   
print(f"Name: {student.get_name()}")
print(f"Roll No: {student.get_roll_no()}")  
print(f"Marks: {student.get_marks()}")