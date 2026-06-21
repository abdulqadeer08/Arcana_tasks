user_input = 'random'
data = []

def user_menu():
    print ('1. ADD AN ITEM')
    print('2. MARK AS COMPLETE')
    print('3. VIEW THE ITEMS')
    print('4. EXIT')

while user_input != '4':
    user_menu()
    user_input =  input('ENTER YOUR CHOICE NUMBER : '   ) 
    if user_input == '1':
        item = input('what is to be done? :')
        data.append(item)
        print('added item :', item)
    elif user_input == '2':
        item = input('what is to be marked as complete? :')
        if item in data:
            data.remove(item)
            print('MARK AS COMPLETE ')
        else:
            print('ITEM NOT FOUND')
    elif user_input == '3':
        print('LIST OF TO DO ITEMS:')
        for item in data:
            print(item)
        if not data:
            print('NO ITEMS in PENDING')
    elif user_input == '4':
        print('GOOD BYE!')
    else:
        print('PLEASE ENTER 1,2,3 OR 4')
    