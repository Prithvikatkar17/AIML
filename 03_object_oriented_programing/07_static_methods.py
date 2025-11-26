class Laptop:
    storage_type = "SSD"

    def __init__(self, ram, rom):
        self.ram = ram
        self.rom = rom

    @classmethod
    def print_storage_type(cls):
        print("Storage Type:", cls.storage_type)

    @staticmethod
    def greet():
        print("Welcome to the Laptop Store!")

    def print_info(self):
        print("Laptop RAM:", self.ram)
        print("Laptop ROM:", self.rom)
        self.print_storage_type()
        self.greet()
# creating object
l1 = Laptop("8GB", "512GB")
l1.print_info()
Laptop.greet()