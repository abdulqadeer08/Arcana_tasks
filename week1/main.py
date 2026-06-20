
#READING TO A FILE
# f = open('myfile.txt', 'r')
# text = f.read()
# print(text)
# f.close()


#WRITING TO A FILE
# f = open('myfile.txt', 'a')
# f.write('Hello, World!')
# # print(f)
# f.close()


#using readline()
f = open('myfile.txt', 'r')
while True:
    line = f.readline()
    if not line:
        break
    print(line)