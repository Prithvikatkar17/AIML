num=[3,5,7,4,9,8]

x=9
idx=0
for i in num:
    if(i == x):
        print(f"found at index {idx}")
        break
    idx+=1
