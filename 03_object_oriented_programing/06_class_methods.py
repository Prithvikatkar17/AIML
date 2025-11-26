class Laptop:
    storage_type = "SSD"

    def __init__(self, ram, rom):
        self.ram = ram
        self.rom = rom

    @classmethod  #decorator to define class method
    def print_storage_type(cls):
        print("Storage Type:", cls.storage_type)

    def print_info(self):
        print("Laptop RAM:", self.ram)
        print("Laptop ROM:", self.rom)
        self.print_storage_type()
# creating object
l1 = Laptop("8GB", "512GB")
l1.print_info()
# calling class method without creating object
Laptop.print_storage_type()
