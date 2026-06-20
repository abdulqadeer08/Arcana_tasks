# # # this is simple while loop concept.
# # # count = 1
# # # while count <=5:
# # #     print ("Qadeer")
# # #     count += 1


# # # iam printing number from 1 to 5 using while loop.
# # # i = 1
# # # while i <= 5:
# # #     print (i)
# # #     i += 1



# # #PRINTING NUMBER FROM 5 TO 1 USING WHILE LOOP.
# # # i = 5
# # # while i>= 1:
# # #     print (i)
# # #     i -= 1
# # #PRACTICE 

# # #Q1 : PRINT NUMBER FROM 1 TO 1000.
# # # i = 1
# # # while i <= 1000:
# # #         print(i)
# # #         i += 1

# # #         #Q2 : PRINT NUMBER FROM 1000 TO 1.
# # # i = 1000
# # # while i>= 1:
# # #     print(i)
# # #     i -= 1


# # #Q3 : MULTIPLICATION TABLE OF 3.
# # # n = 3
# # # i = 1
# # # while i <= 10:
# # #     print(n,"x", i,"=", n*i)
# # #     i += 1

# # #Q4 : PRINT THE LIST ELEMENTS BY USING LOOP.
# # # idx = 0
# # # list1 = [10,20,30,40,50,60,70,80,90,100]
# # # while idx < len(list1):
# # #     print(list1[idx])
# # #     idx += 1


# # # Q5 : SEARCH A NUMBER IN THE TUPLE.

# # # tuple1 = (10,20,30,40,50,60,70,80,90,100)
# # # i = 0
# # # x = 90
# # # while i < len(tuple1):
# # #     if (tuple1[i] == x):
# # #         print("number found at index", i)
# # #     i += 1






# # # FOR LOOP #
# # # PRINT A NUMBER OF LIST BY USING FOR LOOP 
# # # list1 = [10,20,30,50]
# # # for val in list1:
# # #     print(val)


# # #Q1: PRINT THE LIST ELEMENTS BY USING FOR LOOP.
# # list1 = [10,20,30,40,50,60,70,]
# # for val in list1:
# #     print(val)



# # Q2: SEARCH FOR X NUMBER IN THE TUPLE BY USING FOR LOOP.
# # tuple1 = (10,20,30,40,50,60,70,80,90,100)
# # x = 80
# # idx = 0
# # for val in tuple1:
# #     if(val == x):
# #         print("FOUND AT IDX",idx)
# #     idx += 1




# # #FIND THE SUM OF FIRST N NUMBERS.
# # n = 5
# # sum = 0
# # for i in range(1, n+1):
# #     sum += i
# # print("Sum of first", n, "numbers is:", sum) 


#  #FIND THE SUM OF FIRST N NUMBERS BY USING WHILE LOOP.
# n = 5
# sum = 0
# i = 1
# while i <= n:
#     sum += i
#     i += 1 
#     print (sum)


# FIND THE FACTORIAL OF A NUMBER BY USING WHILE LOOP.
n = 5
fac = 1
i = 1
while i <= n:
    fac *= i
    i += 1
    print (fac)