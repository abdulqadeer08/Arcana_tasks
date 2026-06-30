# # ---------------------------------------------------------------------------------------------  
#         #Task 1.1: Create a NumPy array representing hourly temperature readings for a day.
# # ---------------------------------------------------------------------------------------------  

# import numpy as np

# celcius_arr = np.array([16.5, 16.0, 15.8, 15.5, 15.2, 15.7, 16.8, 18.2, 
#     19.5, 21.3, 23.0, 24.8, 26.2, 26.5, 26.0, 25.1, 
#     24.0, 22.7, 21.1, 19.8, 18.9, 18.1, 17.5, 17.0
# ])
# # Find min, max, and average temperature  
# avg = celcius_arr.mean()
# min = celcius_arr.min()
# max = celcius_arr.max()

# # Convert Celsius readings to Fahrenheit.
# fahrenheit_temps = (celcius_arr * 9/5) + 32

# # Display results
# print(f"MAX TEMP  :, {max}°C")
# print(f"MIN TEMP  : , {min}°C")
# print(f"AVG TEMP  : ,{avg}°C")
# print(np.round(fahrenheit_temps, 1))


#-----------------------------------------------------------------------------------------------------------------------------------
#Task 2.1: Create an array of 1000 random customer ages (between 18 and 60).Find the percentage of customers in the age group 25–35.
#-----------------------------------------------------------------------------------------------------------------------------------
# import numpy as np
# customer_ages = np.random.randint(18, 60, size = 1000)
# group = (customer_ages >= 25) & (customer_ages <= 35)
# per = (np.sum(group)/ len(customer_ages)) * 100 

# print(f"TOTAL CUSTOMER LIES IN 25 TO 25 ARE :{np.sum(group)}")
# print(f"TOTAL PERCTAGEG THOSE LIE IN 25 TO 35 :{per}%")


#-----------------------------------------------------------------------------------------------------------------------------------
#                        Task 3.1: Given daily stock prices for a company in a NumPy array:
#-----------------------------------------------------------------------------------------------------------------------------------


# import numpy as np
# #  prices = np.array([120, 121, 119, 122, 118, 117, 115, 116, 118, 120])
# prices = np.array([120,121,119,122,118,117,115,116,118,120])

# #  Extract prices for days 2–6.
# days_2_to_6 = prices[1:6]

# #  Get all prices greater than 118.
# greater_than_118 = prices[prices > 118 ]

# #DISPALY
# print ("PRICES FOR DAYS 2-6 ARE : ", days_2_to_6)
# print("PRICES GREATER THAN 118 ARE : ",greater_than_118)





#------------------------------------------------------------------------------------------------------------------------
#     Task 4.1:
#     Create a 3×3 matrix of employee salaries in thousands.Add a bonus vector [5, 10, 15] to each row using broadcasting.
#------------------------------------------------------------------------------------------------------------------------

# import numpy as np
# salaries = np.random.random_integers(10000,200000, size=(3,3))
# # print (salaries)
# bonus_vector = np.array([5,10,15])

# add = salaries + bonus_vector
# print ("FINAL RESULT IS :",add)




#--------------------------------------------------------------------------------------------------------------------
#                           Task 5.1: You have monthly sales for 3 years in a 1D array (length 36).
#--------------------------------------------------------------------------------------------------------------------
import numpy as np 
monthly_sales = np.random.randint(100, 500, size = 36)
# print ("ORIGINAL 1D ARRAY",monthly_sales)

#             *) Reshape into a 3×12 matrix (3 years, 12 months)
reshape_1D = monthly_sales.reshape(3, 12)


#             *) Find total sales per year
total_sales_year = reshape_1D.sum(axis=1)
# print (total_sales_year)

#             *) Find the month with the highest average sales across years.
avg = reshape_1D.mean(axis = 0)
best_month = np.argmax(avg)
print (best_month)