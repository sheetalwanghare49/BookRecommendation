#import the libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

#read the data
books = pd.read_csv('Books (3).csv',encoding='latin1')
users = pd.read_csv('Users (2).csv',encoding='latin1',on_bad_lines='skip')
rating = pd.read_csv('Ratings (2).csv',encoding='latin1')

#fill that column of rating
rating['Book-Rating'].fillna(rating['Book-Rating'].mean(),inplace=True)

rating['Book-Rating'] = rating['Book-Rating'].astype(int)

x = rating['User-ID'].value_counts()>200
y = x[x].index
rating = rating[rating['User-ID'].isin(y)]

#here we combine the books data to ratings data
rating_books = rating.merge(books, on='ISBN')

#here we calculate total rating w.r.t book
total_ratings = rating_books.groupby('Book-Title')['Book-Rating'].count().reset_index()

#Rename the column 'Book-Rating' column to 'Number_of_Ratings'
total_ratings.rename(columns={'Book-Rating': 'Number_of_Ratings'},inplace=True)

#here we combine the number_of_ratings to the rating_books
final_ratings = rating_books.merge(total_ratings,on='Book-Title')

final_ratings = final_ratings[final_ratings['Number_of_Ratings']>=50]

#means here final_ratings is our main dataset
book_pivot = final_ratings.pivot_table(columns='User-ID',index='Book-Title',values='Book-Rating')
book_pivot.fillna(0,inplace=True)

#for large or sparse data use cosine similarity
similarity_scores = cosine_similarity(book_pivot)

pickle.dump(book_pivot,open('book_pivot_table.pkl','wb'))
pickle.dump(final_ratings,open('final_data.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_scores,open('similarity_scores.pkl','wb'))

print('Trained sucessfully')