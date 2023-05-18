#Content-based kdramas recommendation web app in streamlit


# Fetch the neccesary python modules
import streamlit as st
import pickle
import pandas as pd
import requests
# import librairies
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math as math
import time 
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [14,14]
import scipy as sp
import itertools
import streamlit.components.v1 as components
from pyvis.network import Network




# Recommend movies based on content similarity (cosine similarity model)
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_poster = []

    #Fetch the posters for each recommended movie
    for  i in movies_list:
        movie_id = movies.iloc[i[0]].ID
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_poster.append(movies["image_URL"].iloc[movie_id])

    return recommended_movies,recommended_movies_poster

#netwrokx graph based model
#importing necessary libraries
import networkx as nx
import itertools
G = pickle.load(open('graph.pickle', 'rb'))

# Load the necessary python pickle files
movies_dict = pickle.load(open('movie_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl','rb'))




G = pickle.load(open('graph.pickle', 'rb'))

# Load the necessary python pickle files
movies_dict = pickle.load(open('movie_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)

def get_recommendation(root):
    commons_dict = {}
    for e in G.neighbors(root):
        for e2 in G.neighbors(e):
            if e2==root:
                continue
            if G.nodes[e2]['label']=="MOVIE":
                commons = commons_dict.get(G.nodes[e2]['key'])
                if commons==None:
                    commons_dict.update({G.nodes[e2]['key'] : [e]})
                else:
                    commons.append(e)
                    commons_dict.update({G.nodes[e2]['key'] : commons})

    movies_list=[]
    weight=[]
  
    for key, values in commons_dict.items():
        w=0.0
        for e in values:
            w=w+1/math.log(G.degree(e))
        movies_list.append(key) 
        weight.append(w)
    
    #we need this function in order to be able to sort weights after find the closest movies
    result = pd.Series(data=np.array(weight),index=movies_list)
    result.sort_values(inplace=True,ascending=False)   
    
    #create another dictionary in order to save 5 closest ones (it can be 10 as well)
    first_five= dict(itertools.islice(result.items(), 5))
    
    titles_list=[]
    #creating the list of names by id's
    for i in (first_five):
        titles_list.append(movies.iloc[i]["title"])
     
    #to fetch posters from the site using image_URL
    recommended_movies_poster = []
    for  i in first_five:
        recommended_movies_poster.append(movies["image_URL"].iloc[i])

    return titles_list,recommended_movies_poster

#function for creating the graph
def get_all_adj_nodes(list_in):
    sub_graph=set()
    for m in list_in:
        sub_graph.add(m)
        for e in G.neighbors(m):        
                sub_graph.add(e)
    return list(sub_graph)

def draw_sub_graph(sub_graph):
    subgraph = G.subgraph(sub_graph)
   

    for n in subgraph.nodes(data=True):
        n[1]['label']=n[0] #concatenate label of the node with its attribute

    # Initiate PyVis network object
    graph_net = Network( height='600px', width='100%', bgcolor='ffffff')

    # Take Networkx graph and translate it to a PyVis graph format
    graph_net.from_nx(subgraph)
        # Generate network with specific layout settings
    graph_net.repulsion(node_distance=420, central_gravity=0.33,
                       spring_length=110, spring_strength=0.10,
                       damping=0.95)
    
    try:
        path = 'tmp'
        graph_net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

    # Save and read graph as HTML file (locally)
    except:
        path = 'html_files'
        graph_net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

    # Load HTML file in HTML component for display on Streamlit page
    components.html(HtmlFile.read(), height=435)



# Web app's main section
st.title("KDrama Recommender System")

selected_movie_name = st.selectbox(
'Select a kdrama to recommend',
movies['title'].values)

# Output recommendations with posters
if st.button('Recommend'):
    name, posters = get_recommendation(selected_movie_name)
    col1, col2, col3, col4,  col5 = st.columns(5)
    with col1:
        st.text(name[0])
        st.image(posters[0])
    with col2:
        st.text(name[1])
        st.image(posters[1])
    with col3:
        st.text(name[2])
        st.image(posters[2])
    with col4:
        st.text(name[3])
        st.image(posters[3])
    with col5:
        st.text(name[4])
        st.image(posters[4])
    reco=list(name)
    reco.extend(selected_movie_name.split("."))
    sub_graph = get_all_adj_nodes(reco)
    draw_sub_graph(sub_graph)

page_bg_img = '''
<style>
body {
background: linear-gradient(315deg, #4f2991 3%, #7dc4ff 38%, #36cfcc 68%, #a92ed3 98%);
animation: gradient 15s ease infinite;;
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
