class product:
    def __init__(self, name, price, ):
        self.name = name
        self.price = price

    def get_info(self):
        print("Product Name:", self.name)
        print("Product Price:", self.price)
# creating object
p1 = product("Laptop", 50000)
p2 = product("Smartphone", 20000)
# calling method
p1.get_info()
p2.get_info()