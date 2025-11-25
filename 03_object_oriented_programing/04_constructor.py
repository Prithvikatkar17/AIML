class student:
    def __init__(self):# constructor
        print("This is constructor") # constructor method

s1= student() # creating object
s2= student()

class Student:
    def __init__(self, name, age): # parameterized constructor
        self.name = name
        self.age = age

    def show(self):
        print("Name:", self.name)
        print("Age:", self.age)
# creating object
s1 = Student("Pruthviraj", 20)
# calling method
s1.show()
s2 = Student("Alice", 22)
s2.show()