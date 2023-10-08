#!/usr/bin/env python
# coding: utf-8

# 1. Display Fibonacci Series upto 10 terms
# 2. Display numbers at the odd indices of a list
# 3. Print a list in reverse order
# 4. Your task is to count the number of different words in this text
# 	
# 	string = """
# 	ChatGPT has created this text to provide tips on creating interesting paragraphs. 
# 	First, start with a clear topic sentence that introduces the main idea. 
# 	Then, support the topic sentence with specific details, examples, and evidence.
# 	Vary the sentence length and structure to keep the reader engaged.
# 	Finally, end with a strong concluding sentence that summarizes the main points.
# 	Remember, practice makes perfect!
# 	"""
# 
# 5. Write a function that takes a word as an argument and returns the number of vowels in the word
# 6. Iterate through the following list of animals and print each one in all caps.
# 
#   animals=['tiger', 'elephant', 'monkey', 'zebra', 'panther']
# 
# 7. Iterate from 1 to 15, printing whether the number is odd or even
# 8. Take two integers as input from user and return the sum

# In[9]:


#1 Display Fibonacci Series upto 10 terms

first_number = 0
second_number = 1  

i = 0

while i <= 10:
    if i <= 0:
        print('the fibbonacchi number 1 is',first_number)
        i = i+1
    else:
        fibbonacci_number = first_number + second_number
        print('the fibbonacci number',i+1,"is",fibbonacci_number)
        
        first_number = second_number
        second_number = fibbonacci_number
        i = i+1


# In[1]:


#2.Display numbers at the odd indices of a list
List1 = [0,1,2,3,4,5,6,7,8,9,10]

for i in List1:
    #odd indices when divided by 2 the remainder will not be equal to zero.
        if i%2 != 0:
            print("the number at odd indice",i,"is",i)


# In[2]:


#3.Print a list in reverse order
Numbers = [1,2,3,4,5,6,7,8,9]

reverse_list = Numbers[-1:-10:-1]

print(reverse_list)


# In[3]:


#4Your task is to count the number of different words in this text

#string = """ ChatGPT has created this text to provide tips on creating interesting paragraphs. First, start with a clear topic sentence that introduces the main idea. Then, support the topic sentence with specific details, examples, and evidence. Vary the sentence length and structure to keep the reader engaged. Finally, end with a strong concluding sentence that summarizes the main points. Remember, practice makes perfect! """
string = "ChatGPT has created this text to provide tips on creating interesting paragraphs. First, start with a clear topic sentence that introduces the main idea. Then, support the topic sentence with specific details, examples, and evidence. Vary the sentence length and structure to keep the reader engaged. Finally, end with a strong concluding sentence that summarizes the main points. Remember, practice makes perfect!"
string_list = string.split(" ")

count = 0

for i in string_list:
    count = count + 1
print("number of words in the string:",count)

different_words_count = 0

for x in range(62):
    temp = string_list[x]
    repeat = 0
    
    #iterating over the previous elements in the list to check whether there is any repeated element
    for z in range(x):
        if temp == string_list[z]:
            repeat = 1
    if repeat == 1:
        different_words_count = different_words_count
    else:
        different_words_count = different_words_count + 1
print("number of different words:",different_words_count)   


# In[4]:


#5Write a function that takes a word as an argument and returns the number of vowels in the word
def vowel_counter(a):
    vowel = 0
    word_list = list(a)
    for letter in word_list:
        if letter.lower() == "a":
            vowel = vowel+1
        elif letter.lower() == "e":
            vowel = vowel+1
        elif letter.lower() == "i":
            vowel = vowel+1
        elif letter.lower() == "o":
            vowel = vowel+1     
        elif letter.lower() == "u":
            vowel = vowel+1
    return(vowel)


# In[5]:


vowel_counter("hello")


# In[6]:


#6Iterate through the following list of animals and print each one in all caps.

#animals=['tiger', 'elephant', 'monkey', 'zebra', 'panther']

animals=['tiger', 'elephant', 'monkey', 'zebra', 'panther']

for animal in animals:
    print(animal.upper())


# In[7]:


#7.Iterate from 1 to 15, printing whether the number is odd or even

for i in range(1,16):
    if i%2 == 0:
        print (" the number",i,"is an even integer")
    else:
        print("the number",i,"is an odd integer")


# In[8]:


#8.Take two integers as input from user and return the sum

a = input("please provide input for the fist integer:")
b = input("please provide input for the second integer:")

total = int(a) + int(b)

print("the sum of two integers is :",total)

