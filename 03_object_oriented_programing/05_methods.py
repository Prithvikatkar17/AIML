class laptop:
    storage_type = "SSD"

    def __init__(self,ram,rom):
        self.ram = ram
        self.rom = rom
    def print_info(self):
        print("Laptop RAM:", self.ram)
        print("Laptop ROM:", self.rom)
        print("Storage Type:", self.storage_type)

# creating object
l1 = laptop("8GB", "512GB")
l1.print_info()