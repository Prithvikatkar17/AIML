class enplayee:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def display_info(self):
        return f"Name: {self.name}, Age: {self.age}"
    

class teacher(enplayee):
    def __init__(self, name, age, subject):
        super().__init__(name, age)
        self.subject = subject

    def display_info(self):
        base_info = super().display_info()
        return f"{base_info}, Subject: {self.subject}"
    
t1 = teacher("Alice", 30, "Mathematics")
print(t1.display_info())  # Output: Name: Alice, Age: 30, Subject: Mathematics
