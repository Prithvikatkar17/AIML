square =[]

for i in range (6):
    square.append(i**2)

print(square)

Square= [i**2 for i in range(6) if i%2==0]
print(Square)

num = [-2,3,-5,8,1,-9]
num1= [0 if i<0 else i for i in num]
print(num1)