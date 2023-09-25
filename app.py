import model
import data
import faiss
import numpy as np
import time
from pprint import pprint
import streamlit as st

index = faiss.read_index("movie_plot.index")

def fetch_movie_info(dataframe_idx):
    info=data.df.iloc[dataframe_idx]
    meta_dic={}
    meta_dic["movie_name"]=info["movie_name"]
    meta_dic["genre"]=info["genre"]
    return meta_dic

def search(query,top_k,index,model):
    t=time.time()
    
    query_vector=model.model.encode([query])
    top_k=index.search(query_vector,top_k)
    
    print('>>>> Results in Total Time: {}'.format(time.time()-t))
    
    top_k_ids=top_k[1].tolist()[0]
    top_k_ids=list(np.unique(top_k_ids))
    results=[fetch_movie_info(idx) for idx in top_k_ids]
    return results


#query="Artificial Intelligence based action movie"
#results=search(query, top_k=5, index=index, model=model)

#print("\n")
#for result in results:
    #print('\t',result)


# Streamlit app
st.title("Movie Semantic Search")

# User input for movie query
query = st.text_input("Enter you expection:")

if st.button("Search"):
    if query :
        results =search(query, top_k=5, index=index, model=model)

    for result in results:
        st.write('\t',result)
