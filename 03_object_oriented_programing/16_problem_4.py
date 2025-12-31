# Create a shape class with a method area().
#  Create subclasses, triangle,circle and square that override the area() method. 
class Shape:
    def area(self):
        pass

class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def area(self):
        return 0.5 * self.base * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius * self.radius

class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side * self.side
    
# Example usage:
triangle = Triangle(10, 5)
circle = Circle(7)
square = Square(4)
print(f"Triangle area: {triangle.area()}")
print(f"Circle area: {circle.area()}")
print(f"Square area: {square.area()}")