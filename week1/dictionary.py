# qadeer = {
#     "name" : "Qadeer",
#     "CGPA" : 3.4,
#     "marks" : [90,99,100],
# }
# print(qadeer["marks"])


marks ={}

x = int(input("Enter phy: "))
marks.update({"phy": x})

x = int(input("Enter chem: "))
marks.update({"chem": x})

print (marks)

