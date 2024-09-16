import streamlit as st
import pandas as pd
import pickle
import numpy as np

#read the data
final_data = pickle.load(open('final_data.pkl','rb'))
data = pd.DataFrame(final_data)

book_pt = pickle.load(open('book_pivot_table.pkl','rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))

def recommend(book_name):
    # index fetch
    index = np.where(book_pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:6]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = final_data[final_data['Book-Title'] == book_pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Publisher'].values))
        
        data.append(item)
    
    return data

st.title('Book Recommendation System')

book_name = st.selectbox('Type or select Book from the dropdown', [''] + list(data['Book-Title'].unique()))

# Button for triggering recommendation
if st.button('Show Recommendation'):
    if book_name == '':
        st.warning('Please select a book name.')
    else:
        recommended_books = recommend(book_name)
        
        if isinstance(recommended_books, str):
            st.error(recommended_books)  # If book not found, display the error message
        else:
            # Display each recommended book with its image on the left and details on the right
            for book in recommended_books:
                col1, col2 = st.columns([0.8, 2])  # Two columns, image on the left, details on the right
                
                with col1:
                    st.image(book[2], width=150)  # Display book image
                
                with col2: #display details
                    st.subheader(book[0]) 
                    st.text(f"Author: {book[1]}") 
                    st.text(f"Publisher: {book[3]}") 

                    