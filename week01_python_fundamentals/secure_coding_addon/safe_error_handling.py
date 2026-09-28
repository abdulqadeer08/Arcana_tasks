try:
    num = int(input("Enter number: "))
    print(100 / num)

except ValueError:
    print("Please enter numbers only.")

except ZeroDivisionError:
    print("Cannot divide by zero.")