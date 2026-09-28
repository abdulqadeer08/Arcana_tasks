from pathlib import Path

# myfile.txt lives in the week's data/ folder
MY_FILE = Path(__file__).resolve().parent.parent / 'data' / 'myfile.txt'

#READING TO A FILE
# f = open(MY_FILE, 'r')
# text = f.read()
# print(text)
# f.close()


#WRITING TO A FILE
# f = open(MY_FILE, 'a')
# f.write('Hello, World!')
# # print(f)
# f.close()


#using readline()
f = open(MY_FILE, 'r')
while True:
    line = f.readline()
    if not line:
        break
    print(line)