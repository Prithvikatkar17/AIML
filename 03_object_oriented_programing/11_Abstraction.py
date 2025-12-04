from abc import ABC, abstractmethod

class animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass
class dog(animal):
    def make_sound(self):
        return "Bark"
    
d1 = dog()
print(d1.make_sound())

class cat(animal):
    def make_sound(self):
        return "Meow"
c1 = cat()
print(c1.make_sound())