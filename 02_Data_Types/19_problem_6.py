# Given a list of words:
# words =["apple","banana","kiwi","cherry","mango"]
# Create a dictionary that maps each word to its length

words =["apple","banana","kiwi","cherry","mango"]
len_words=len(words)
dict={}

for i in range(len_words-1):
    dict[words[i]]=len(words[i])

print(dict)