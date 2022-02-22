'''
Name: Pruthvik Kakadiya
UTA ID: 1001861545
Machine Learning (CSE-6363)
Programming Assignment 2
Movie Recommendation Using Matrix Factorization

To run:
go to the file where you downloaded the code:
    --> python  Kakadiya_PA2.py

'''

import pandas as pd
import numpy as np
import time

print(".................................................................")
print("Reading the ratings dataset, please wait..")
ratings = pd.read_csv("ml-20m/ratings.csv", skipfooter = 19994500, engine = 'python')
print(".................................................................")
ratings.userId = ratings.userId.astype('category').cat.codes.values
ratings.movieId = ratings.movieId.astype('category').cat.codes.values

index=list(ratings['userId'].unique())
columns=list(ratings['movieId'].unique())
index=sorted(index)
columns=sorted(columns)
df=pd.pivot_table(data=ratings,values='rating',index='userId',columns='movieId').fillna(0)


def matrix_factorization(Matrix, P, Q, K, iterations=1000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(iterations):
        for i in range(len(Matrix)):
            for j in range(len(Matrix[i])):
                if Matrix[i][j] > 0:
                    element_ij = Matrix[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * element_ij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * element_ij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        error = 0
        for i in range(len(Matrix)):
            for j in range(len(Matrix[i])):
                if Matrix[i][j] > 0:
                    error = error + pow(Matrix[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        error = error + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if error < 0.001:
            break
    print("Matrix Factorization Completed")
    return P, Q.T

def get_recommendation (df, user, n=10):
  a = df.to_numpy()[0][user:]
  l = sorted(range(len(a)), key=lambda i: a[i])[-1*n:]
  movies_list = []
  for i in l:
    movies_list.append([movies['title'][i], movies['genres'][i]])
  return movies_list


start_time = time.time()
user = int(input("Enter UserId:"))
count = int(input("How many movies to recommend:"))
print(".................................................................")
ratings_array = np.array(df)
N = len(ratings_array)
M= len(ratings_array[0])
K = 2
P = np.random.rand(N,K)
Q = np.random.rand(M,K)
print("Matrix Factorization in progress, please wait..")
new_P, new_Q = matrix_factorization(ratings_array, P, Q, K)
new_prediction_matrix = np.dot(new_P, new_Q.T)

pd.DataFrame(new_prediction_matrix).to_csv('predictions.csv')
pred = pd.read_csv('predictions.csv')
movies = pd.read_csv("ml-20m/movies.csv")

print(".................................................................")
print("Recommendation from non-rated movies.")
recommendation = get_recommendation(df= pred, user= user, n= count)
for s in recommendation:
    print(*s)
print("Execution Time: %s seconds" % (time.time() - start_time))