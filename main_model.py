# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:13:29 2024

@author: faranak
"""

import math
import networkx as nx
import matplotlib.pyplot as plt

# Define clusters of clients with interconnections within clusters
clusters = {
    "Cluster 1": ["Client 1", "Client 2", "Client 3", "Client 4"],
    "Cluster 2": ["Client 5", "Client 6", "Client 7", "Client 8"],
    "Cluster 3": ["Client 9", "Client 10", "Client 11", "Client 12"]
}
server = "Server"
leaders = ["Leader 1", "Leader 2", "Leader 3"]

# Create a graph
G = nx.DiGraph()

# Add server node
G.add_node(server)

# Define positions of nodes manually for better visualization
pos = {}

# Define positions for clients in clusters in circular shapes
radius = 2
angle_step = 360 / 4  # 4 clients per cluster

for cluster_id, (cluster_name, clients) in enumerate(clusters.items(), start=1):
    center_x, center_y = (cluster_id - 1) * 6, 0  # space out clusters horizontally
    for i, client in enumerate(clients):
        angle = math.radians(angle_step * i)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        pos[client] = (x, y)
        G.add_node(client)

    # Add leader node for the cluster
    leader = leaders.pop(0)
    G.add_node(leader)
    pos[leader] = (center_x, center_y - radius - 1)  # place leader below the cluster
    
    # Connect clients within the cluster (circular connections)
    for i in range(len(clients)):
        G.add_edge(clients[i], clients[(i + 1) % len(clients)])  # Ring topology within the cluster

    # Connect clients to the leader
    for client in clients:
        G.add_edge(client, leader)  # Clients send updates to the leader

    # Connect leader to the server
    G.add_edge(leader, server)  # Leader sends updates to the server
    G.add_edge(server, leader)  # Server sends the updated model to the leader



# Adjust server position to be lower
pos[server] = (8, -5)

plt.figure(figsize=(14, 10))

# Draw nodes with different colors for clients, leaders, and server
colors = ["lightblue", "lightgreen", "lightcoral"]
for i, cluster in enumerate(clusters.values()):
    nx.draw_networkx_nodes(G, pos, nodelist=cluster, node_color=colors[i], node_size=2000, label=f"Cluster {i + 1} Clients")
nx.draw_networkx_nodes(G, pos, nodelist=["Leader 1", "Leader 2", "Leader 3"], node_color="orange", node_size=3000, label="Leaders")
nx.draw_networkx_nodes(G, pos, nodelist=[server], node_color="yellow", node_size=4000, label="Server")

# Draw edges
nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color="gray")

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

# Display the model diagram
plt.title("Clustered Adaptive Ring Federated Learning (CARFL) Model with Circular Clusters and Leaders", fontsize=14)
plt.legend(scatterpoints=1, loc="center left", bbox_to_anchor=(0.9, 0.9))
plt.show()
