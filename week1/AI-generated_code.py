# import os
# import mysql.connector
# from dotenv import load_dotenv

# load_dotenv()

# API_KEY = "12345-SECRET-KEY"
# DB_PASSWORD = "admin123"

# def connect_db():
#     conn = mysql.connector.connect(
#         host="localhost",
#         user="root",
#         password=DB_PASSWORD,
#         database="users_db"
#     )
#     return conn

# def get_user_data():
#     user_id = input("Enter user id: ")

#     conn = connect_db()
#     cursor = conn.cursor()

#     query = f"SELECT * FROM users WHERE id = {user_id}"
#     cursor.execute(query)

#     data = cursor.fetchall()

#     return data

# def process_data():
#     try:
#         data = get_user_data()
#         print("User Data:", data)

#     except:
#         pass

# process_data()





#AFTER FIXING PROBLEM BELOW CODE IS SAFE TO RUN.

import os
from dotenv import load_dotenv

load_dotenv()


def get_age():
    try:
        age = int(input("Enter your age: "))
        if age < 0 or age > 120:
            print("your not eligible to enter this site.")
        else:
            print("Your age is:", age)
    except ValueError:
        print("Please enter a valid number for age.")

def divide_numbers():
    try:
        a = int(input("Enter first number: "))   
        b = int(input("Enter second number: "))

        result = a / b
        print("Result:", result)

    except ValueError:
        print("Please enter valid numbers.")
    except ZeroDivisionError:
        print("Cannot divide by zero.")

def save_user():
    name = str(input("Enter name: ")).strip()
    if not name:
        print("Name cannot be empty.")
        return
    try:
        with open("users.txt", "a") as file:
            file.write("Name: " + name + "\n")
    except Exception as e:
        print("Error occurred while saving user data:", e)

get_age()
divide_numbers()
save_user()
