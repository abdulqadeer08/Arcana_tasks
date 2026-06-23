
# # # import matplotlib.pyplot as plt

# # # x = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sund']
# # # y = [35, 20, 60, 75, 4, 10, 55]

# # # plt.plot(x,y, color =  'red', linestyle = '--', linewidth = 3, marker = 's', label = '2025 sales data')
# # # plt.title('SALE OF THE SHOP 2025')
# # # plt.xlabel('Sales of the week')
# # # plt.ylabel('Sale of the day')
# # # plt.legend(loc = 'upper left', fontsize = 10)
# # # plt.grid(color = 'gray', linestyle = ':', linewidth = 1)
# # # # plt.xlim(1,4)
# # # # plt.ylim(0, 2000)
# # # plt.xticks(['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sund'], ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7'])

# # # plt.show()




# ------------------------------------------------------------------------------
#                                   BAR CHART PRACTICE
# -------------------------------------------------------------------------------
# import matplotlib.pyplot as plt
# product= ['A','B','C','D','E']
# sales  = [1000, 1200, 1000, 1500, 800]
# plt.bar (product,sales, color = 'orange', linewidth = 2, label = '2025 sales')
# plt.xlabel('PRODCUT')
# plt.ylabel('SALES')
# plt.title('PRODCUT COMPARSION SALE')
# plt.legend()
# plt.show()






# # ------------------------------------------------------------------------------
# #                                  PIE CHART PRACTICE
# # -------------------------------------------------------------------------------
# import matplotlib.pyplot as plt
# regions = ['NORTH', 'SOUTH', 'WEST', 'EAST']
# revenue =  [10000, 20000, 30000, 40000]
# plt.pie(revenue, labels = regions, colors = ['gold', 'skyblue', 'coral', 'lightgreen'], autopct = '1.1%ff%%')
# plt.title('REVENUE CONTRIBUTION BY REGIONS')
# plt.show()








# # ------------------------------------------------------------------------------
# #                                 PIE CHART PRACTICE
# # -------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# score = [10,89,76,53,22,41,10,90,89,44,22,51,60,46]
# plt.hist(score, bins = 10, color = 'purple', edgecolor = 'black' )
# plt.xlabel('SCORE RANGES')
# plt.ylabel('NUMBER OF STUDNETS')
# plt.title('SCORE DISTRIBUTION OF DATA')
# plt.show()









#-----------------------------------------------------------------------------------
#                                  SCATTER PLOT PRACTICE
#----------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# hours = [1,2,3,4,5,6,7]
# score = [10, 20, 30, 40, 50, 60, 70]
# plt.scatter(hours, score, color = 'green', marker = 'o', label = 'Student data')
# plt.xlabel('STUDY HOURS')
# plt.ylabel('SCORE OF A STUDENT')
# plt.title('RELATIONSHIP BETWEEN STUDY HOURS AND SCORE')
# plt.grid(True)
# plt.show()





#----------------------------------------------------------------------------------
#                              TWO GROUPS - SCATTER PLOT PRACTICE
#----------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# plt.scatter ([1,2,3,4], [10,20,30,40],color = 'blue', label=' CLASS A') #GROUP 1
# plt.scatter ([1,2,3,4], [30,45,60,10],color = 'gray', label='ClASS B')  #GROUP 2

# plt.xlabel('HOURS STUDIES')
# plt.ylabel('SCORE')
# plt.title('COMPARSION OF TWO GROUPS')
# plt.grid(True)
# plt.legend()
# plt.show()





#----------------------------------------------------------------------------------
#                             SUBPLOT PRACTICE
#----------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# x = [1,2,3,4]
# y = [10,20,15,25]

# plt.subplot(1,2,1)
# plt.plot(x,y)
# plt.title('LINE CHART' )

# plt.subplot(1,2,2)
# plt.bar(x,y)
# plt.title('BAR CHART')
# plt.tight_layout()
# plt.show()





#----------------------------------------------------------------------------------
#                             HOW TO SAVE THESE IMAGES
#----------------------------------------------------------------------------------
#savefig('filename.extension', dpi = value , bbox_inches = 'tight')


# import matplotlib.pyplot as plt
# x = [1,2,3,4]
# y = [10,20,15,25]

# plt.subplot(1,2,1)
# plt.plot(x,y)
# plt.title('LINE CHART' )

# plt.subplot(1,2,2)
# plt.bar(x,y)
# plt.title('BAR CHART')
# plt.tight_layout()
# plt.savefig('visualize.png', dpi = 300, bbox_inches = 'tight')
# plt.show()











# --------------------------------------------------------------------------
                    # WORKING ON NETFLIX DATASET
#---------------------------------------------------------------------------

# import pandas as pd 
# import matplotlib.pyplot as plt

# df = pd.read_csv('netflix_titles.csv')   #load the data
# df = df.dropna(subset= ['type', 'release_year', 'rating', 'country', 'duration']) #clean the data
# type_counts = df['type'].value_counts()
# plt.figure (figsize=(6,4))
# plt.bar(type_counts.index, type_counts.values,  color = ['skyblue', 'orange'])
# plt.title('NUMBER OF MOVIES VS TV SHOW ')
# plt.xlabel('Type')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.savefig('movies_tv.png')
# plt.show()









# ----------------------------------------------------------------
            # RATING OF NETFLIX MOVIES VS SHOW 
#-----------------------------------------------------------------

import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('netflix_titles.csv')
rating_count = df['rating'].value_counts()
plt.figure(figsize=(8,6))
plt.pie(rating_count, labels = rating_count.index, autopct = '%1.1f%%')
plt.title('PERCENTAGE OF CONTENT RATINGS')
plt.tight_layout()
plt.show()