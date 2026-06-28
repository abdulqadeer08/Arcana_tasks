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


