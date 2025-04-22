import streamlit as st
from src.mqlib_utils import MQLibInstance, metadata_df
from src.decomposition import BinaryDecomposition, ExponentialDecomposition
import networkx as nx
import matplotlib.pyplot as plt
from src.sparsification import graph_sparsification_by_effective_resistances


def star_decomposition(G):
    """
    Decomposes an unweighted graph into stars.

    Parameters:
    - G: A NetworkX graph.

    Returns:
    - stars: A list of stars, where each star is represented by its center and neighbors.
    """

    stars = []
    G_copy = G.copy()

    while len(G_copy.edges()) > 0:
        # Sort nodes by highest degree
        node = max(G_copy.nodes(), key=lambda n: G_copy.degree(n))
        neighbors = list(G_copy.neighbors(node))

        # Create a star with the node as center and its neighbors
        star = nx.Graph()
        star.add_node(node)
        for neighbor in neighbors:
            star.add_edge(node, neighbor)
        stars.append(star)

        # Remove the star from the original graph
        G_copy.remove_node(node)

    return stars


def draw_graph_with_edge_weights(G, pos, fig, ax, edge_color='black', max_weight=1.0):
    """
    Draws a graph with edge weights normalized to a maximum weight.

    Parameters:
    - G: A NetworkX graph.
    - max_weight: The maximum weight for normalization.
    """
    if any("weight" in data for _, _, data in G.edges(data=True)):
        # st.write(list(G.edges(data=True)))
        edge_labels = nx.get_edge_attributes(G, "weight")
    else:
        # print(list(G.edges(data=True)))
        edge_labels = {edge: 1 for edge in G.edges()}
    edge_labels = {k: v / max_weight for k, v in edge_labels.items()}

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=5, node_color="blue", ax=ax)
    nx.draw_networkx_edges(G, pos, width=[2*weight for weight in edge_labels.values()], alpha=0.8, ax=ax, edge_color=edge_color)

    plt.box(False)


# Set Streamlit page configuration
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Set the title of the app, center, and style it
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
    }
    </style>
    <div class="title">Union-of-Stars Visualization</div>
    <br>
    """,
    unsafe_allow_html=True,
)

metadata_df = metadata_df[
    (metadata_df['Number of vertices'] >= 5) &
    (metadata_df['Number of vertices'] <= 100) &
    (metadata_df['Positive weighted']) &
    (metadata_df['Aspect ratio'] >= 1.5)
    ]
metadata_df.index = metadata_df.index.astype(str) + " (" + metadata_df['Number of vertices'].astype(str) + " vertices)"

instance_names = list(metadata_df.index.unique())

with st.sidebar:
    st.sidebar.markdown(
        '''
            This app visualizes the [union-of-stars](https://arxiv.org/pdf/2406.14330) construction for weighted graphs
             in the MQLib library. The union-of-stars algorithm uses the graph sparsification and decomposition 
             subroutines, and we allow you to choose the corresponding error parameters for these. We also let you 
             choose the decomposition method, either exponential or binary.
        '''
    )

    instance_name = st.selectbox(
        "Choose an MQLib instance to visualize",
        instance_names,
        index=instance_names.index('g000761 (21 vertices)'),
    )
    instance_name = instance_name.split(" (")[0]

    decomposition = st.selectbox(
        "Choose a decomposition method",
        ["Exponential Decomposition", "Binary Decomposition"],
    )
    decomposition_type = 'exponential' if decomposition == "Exponential Decomposition" else 'binary'

    with st.container(border=True):
        decomposition_epsilon = st.slider(
            "Choose the decomposition epsilon",
            min_value=0.01,
            max_value=5.00,
            value=0.50,
            step=0.01,
        )
    with st.container(border=True):
        sparsification_epsilon = st.slider(
            "Choose the sparsification epsilon",
            min_value=0.01,
            max_value=5.0,
            value=0.50,
            step=0.01,
        )

    st.caption(
        '''
        The union-of-stars [algorithm](https://arxiv.org/abs/2011.08165) was developed by 
         Joel Rajakumar, Jai Moondra, Bryan Gard, Swati Gupta, and Creston D. Herold, with [follow-up](https://arxiv.org/abs/2406.14330)
         for weighted graphs by Jai Moondra, Philip C. Lotshaw, Greg Mohler, and Swati Gupta. The [algorithm](https://arxiv.org/abs/0803.0929) 
         for graph sparsification using effective resistances was developed by Daniel A. Spielman and Nikhil Srivastava.
         
         Contact Jai Moondra (<a href="mailto:jmoondra3@gatech.edu">jmoondra3@gatech.edu</a>) for suggestions or questions. Apache 2.0 licensed.
        ''', unsafe_allow_html=True,
    )

# Load the selected instance
instance = MQLibInstance(instance_name)
G = instance.graph
pos = nx.spring_layout(G, seed=42)
if any("weight" in data for _, _, data in G.edges(data=True)):
    max_weight = max([weight for _, _, weight in G.edges(data="weight")])
else:
    max_weight = 1.5

H = graph_sparsification_by_effective_resistances(G=G, epsilon=sparsification_epsilon, seed=42)
if decomposition == "Binary Decomposition":
    decomposition = BinaryDecomposition(
        graph=H, epsilon=decomposition_epsilon,
    )
else:
    decomposition = ExponentialDecomposition(
        graph=H, epsilon=decomposition_epsilon,
    )


stars = []

for i in range(len(decomposition.constituent_graphs)):
    graph = decomposition.constituent_graphs[i]
    weight = decomposition.constituent_coefficients[i]

    stars_constituent = star_decomposition(graph)
    for star in stars_constituent:
        for u, v in star.edges():
            star[u][v]["weight"] = weight

        stars.append([star, i % 8])


step = st.slider("Step", 0, len(stars), 1)


cols = st.columns(2)

with cols[0]:
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_graph_with_edge_weights(G=G, pos=pos, max_weight=max_weight, fig=fig, ax=ax)

    x_min = min(pos[node][0] for node in pos)
    x_max = max(pos[node][0] for node in pos)
    y_min = min(pos[node][1] for node in pos)
    y_max = max(pos[node][1] for node in pos)

    ax.set_xlim(x_min - 0.1, x_max + 0.1)
    ax.set_ylim(y_min - 0.1, y_max + 0.1)

    st.pyplot(fig)


colors = plt.get_cmap("tab10")

with cols[1]:
    N = len(decomposition.constituent_graphs)
    fig, ax = plt.subplots(figsize=(10, 10))

    # Find boundaries
    x_min = min(pos[node][0] for node in pos)
    x_max = max(pos[node][0] for node in pos)
    y_min = min(pos[node][1] for node in pos)
    y_max = max(pos[node][1] for node in pos)

    G_current = nx.Graph()
    for i in range(step):
        draw_graph_with_edge_weights(G=stars[i][0], pos=pos, max_weight=max_weight, fig=fig, ax=ax, edge_color=colors(stars[i][1]))

    # Set the x and y limits to the boundaries
    ax.set_xlim(x_min - 0.1, x_max + 0.1)
    ax.set_ylim(y_min - 0.1, y_max + 0.1)

    st.pyplot(fig)

