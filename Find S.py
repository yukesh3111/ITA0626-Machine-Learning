traing_data=[['Sunny', 'Warm', 'Normal', 'Strong', 'Y'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Y'],
    ['Rainy', 'Cold', 'High', 'Weak', 'N'],
    ['Sunny', 'Warm', 'High', 'Weak', 'Y']]
result=[]
specific_hy=traing_data[0][:-1]
for i in traing_data:
     if(i[-1]=='Y'):
        for j in range (len(specific_hy)):
            if(specific_hy[j]!=i[j]):
                specific_hy[j]="?"
print(specific_hy)
