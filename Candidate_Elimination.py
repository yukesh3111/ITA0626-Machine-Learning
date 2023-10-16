traing_data=[['Sunny', 'Warm', 'Normal', 'Strong', 'Y'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Y'],
    ['Rainy', 'Cold', 'High', 'Weak', 'N'],
    ['Sunny', 'Warm', 'High', 'Weak', 'Y']]
result=[]
specific_hy=traing_data[0][:-1]
S=[["@","@","@","@"],["@","@","@","@"],["@","@","@","@"],["@","@","@","@"]]
for i in traing_data:
     if(i[-1]=='Y'):
        for j in range (len(specific_hy)):
            if(specific_hy[j]!=i[j]):
                specific_hy[j]="?"
     else:
         for k in range(len(specific_hy)):
             if(specific_hy[k]!=i[k]):
                 S[k][k]=specific_hy[k]

for l in range(len(specific_hy)):
    if(specific_hy[l]=="?"):
        S[l][l]="@"
                 
#print(result)
print(specific_hy)
print(S)
