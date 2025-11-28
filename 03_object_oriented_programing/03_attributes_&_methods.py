class Student:
    # attribute
    def __init__(self, name, age):
        self.name = name   # attribute
        self.age = age     # attribute

    # method
    def show(self):
        print("Name:", self.name)
        print("Age:", self.age)


# creating object
s1 = Student("Pruthviraj", 20)

# calling method
s1.show()
