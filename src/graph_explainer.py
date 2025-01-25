import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.explain import Explainer, GNNExplainer
from tqdm import tqdm
# Load the AIDS dataset
dataset = TUDataset(root='/tmp/AIDS', name='AIDS', use_node_attr=True)

import json

class Converter:
    
    def __init__(self) -> None:
        pass

    def from_graph_to_json(self, graph):
                
        infos: dict = {"x": graph.x.tolist(),
                       "adj":graph.adj.tolist(), 
                       "y":graph.y.tolist()}

        return infos
    
    def convert_graphs_list_to_json(self, graphs_list, name: str = "factual"):
        import json
        
        graphs_dict: dict = {}
        
        for idx, graph in enumerate(graphs_list):
            
            graphs_dict[idx] = self.from_graph_to_json(graph=graph)
            
        with open(f'data/{name}.json', 'w', encoding ='utf8') as json_file:
            
            json.dump(graphs_dict, json_file, ensure_ascii = False)
            
    def from_graphs_to_csv(self, graphs_list, name: str = "factual"):
        import csv
        
        # Open the CSV file for writing
        with open(f'data/{name}_node_data.csv', 'w', newline='') as csvfile:
            
            writer = csv.writer(csvfile)
            
            features_matrix = graphs_list[0].x.tolist()
            
            # Define headers
            num_features = len(features_matrix[0])
            feature_headers = [f'Feature{i+1}' for i in range(num_features)]
            headers = ['GraphID','NodeID'] + feature_headers + ['ConnectedNodes', 'NodeClassification', 'IsTarget']
            writer.writerow(headers)
            
            for idx, graph in enumerate(graphs_list):
                
                adj = graph.adj.tolist()
                features = graph.x.tolist()
                classes = graph.y.tolist()
                target = graph.new_idx if name == "factual" else graph.sub_index

                # Write data for each node
                for i in range(len(adj)):
                    node_id = i  # or i+1 if you prefer node IDs starting from 1
                    
                    # Find connected nodes
                    connected_nodes = [str(j) for j, val in enumerate(adj[i]) if val != 0]
                    connected_nodes_str = ','.join(connected_nodes)
                    
                    is_target = 1 if target == i else 0
                    
                    # Combine all data into a single row
                    row = [idx, node_id] + features[i] + [connected_nodes_str, classes[i], is_target]
                    writer.writerow(row)


def generate_factual_counterfactual_dict_aids(factual_graph, counterfactual_graph, idx):

    # Ensure the graphs have the same number of nodes
    assert factual_graph.x.shape[0] == counterfactual_graph.x.shape[0], "Graphs must have the same number of nodes."
    class_label_map = {0: 'a', 1: 'i'}
    # 1. Identify nodes whose features have changed
    changed_nodes = []
    for node_idx in range(factual_graph.x.shape[0]):
        if not torch.equal(factual_graph.x[node_idx], counterfactual_graph.x[node_idx]):
            changed_nodes.append(node_idx)

    # 2. Identify edges that have changed
    factual_edges = set([tuple(factual_graph.edge_index[:, i].tolist()) for i in range(factual_graph.edge_index.shape[1])])
    counterfactual_edges = set([tuple(counterfactual_graph.edge_index[:, i].tolist()) for i in range(counterfactual_graph.edge_index.shape[1])])

    # Edges that were removed or added
    edges_removed = factual_edges - counterfactual_edges
    edges_added = counterfactual_edges - factual_edges
    changed_edges = list(edges_removed.union(edges_added))
    graph_dict = {
        "Factual_Target_Graph_Class": class_label_map[factual_graph.y.cpu().item()]  ,
        "CounterFactual_Target_Graph_Class": class_label_map[counterfactual.y] ,
        "Node_Differences": changed_nodes,
        "Edge_differences":  changed_edges ,
        "Natural_Language_Explanation": None
    }

    return graph_dict


                
        
def build_natural_language_representation_graph_classification(data):
    # Get node features, edges, and labels from data
    node_features = data.x
    edge_index = data.edge_index
    graph_label = data.y.item() if torch.is_tensor(data.y) else data.y  # Assuming there is a single label for the whole graph

    # Map node label indices to actual chemical symbols
    node_label_map = {
        0: 'C', 1: 'O', 2: 'N', 3: 'Cl', 4: 'F', 5: 'S', 6: 'Se', 7: 'P', 8: 'Na', 9: 'I', 
        10: 'Co', 11: 'Br', 12: 'Li', 13: 'Si', 14: 'Mg', 15: 'Cu', 16: 'As', 17: 'B', 18: 'Pt',
        19: 'Ru', 20: 'K', 21: 'Pd', 22: 'Au', 23: 'Te', 24: 'W', 25: 'Rh', 26: 'Zn', 27: 'Bi',
        28: 'Pb', 29: 'Ge', 30: 'Sb', 31: 'Sn', 32: 'Ga', 33: 'Hg', 34: 'Ho', 35: 'Tl', 36: 'Ni', 37: 'Tb'
    }

    # Map class label indices to class labels
    class_label_map = {0: 'a', 1: 'i'}

    # Create a list to store the node descriptions
    node_descriptions = []

    # Iterate over all nodes in the graph
    for node_idx in range(node_features.shape[0]):
        # Split node features into actual attributes and one-hot encoded labels
        attribute_features = node_features[node_idx, :4]
        label_features = node_features[node_idx, 4:]

        # Extract attributes of the node (chem, charge, x, y)
        attributes_str = []
        attributes_str.append(f"chem = {attribute_features[0]}")
        attributes_str.append(f"charge = {attribute_features[1]}")

        attributes_str.append(f"x = {attribute_features[2]:.4f}")

        attributes_str.append(f"y = {attribute_features[3]:.4f} ")
        
        if len(attributes_str) > 0:
            attributes_str = ", ".join(attributes_str)
        else:
            attributes_str = "no attributes"

        # Extract the one-hot encoded label of the node
        node_label_idx = label_features.argmax().item()  # Get the index of the label
        node_label = node_label_map.get(node_label_idx, "unknown")  # Get the actual chemical symbol

        # Find neighbors of the node from edge_index
        neighbors = edge_index[1, edge_index[0] == node_idx].tolist() + edge_index[0, edge_index[1] == node_idx].tolist()
        neighbors = list(set(neighbors))  # Remove duplicates

        if neighbors:
            neighbors_str = ", ".join([str(neighbor) for neighbor in neighbors])
        else:
            neighbors_str = "no connections"

        # Create description for the node
        node_description = (
            f"The node {node_idx} has as attributes {attributes_str} and is labeled as {node_label}. "
            f"It is connected with the nodes {neighbors_str}."
        )

        # Add the description to the list
        node_descriptions.append(node_description)

    # Create a complete graph description
    graph_description = " ".join(node_descriptions)

    # Map graph label to its corresponding class label
    graph_class_label = class_label_map.get(graph_label, "unknown")

    # Add the graph-level label to the description
    final_description = (
        f"This graph is classified as class {graph_class_label}. The graph contains the following nodes: {graph_description}"
    )

    # Return the final dictionary
    return final_description


# Define a simple GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # Aggregate node embeddings into graph-level embedding
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare dataset
torch.manual_seed(42)
dataset = dataset.shuffle()
train_dataset = dataset[:len(dataset) // 10 * 8]
test_dataset = dataset[len(dataset) // 10 * 8:]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, optimizer, and loss function
model = GCN(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Train the model
for epoch in range(1, 100):
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

# Test the model
model.eval()
correct = 0
for data in test_loader:
    data = data.to(device)
    out = model(data.x, data.edge_index, data.batch)
    pred = out.argmax(dim=1)
    correct += (pred == data.y).sum().item()
print(f'Test Accuracy: {correct / len(test_dataset):.4f}')

# Use GNNExplainer to explain predictions for each graph in the test set
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=1000),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='graph',
        return_type='log_probs',
    ),
)
factual_dict = {}
counterfactual_dict = {}
tot_counterfactuals = 0
key = 0 
f_d = {}
cf_d = {}
# Loop through the test dataset to generate counterfactuals
for idx, data in tqdm(enumerate(test_dataset)):
    data = data.to(device)

    # Get prediction for original graph
    original_out = model(data.x, data.edge_index, data.batch)
    original_pred = original_out.argmax(dim=1).item()

    # Generate explanation using GNNExplainer
    explanation = explainer(data.x, data.edge_index, batch=data.batch)
    print(f"\nGraph {idx + 1}:")
    print("Original Prediction:", original_pred)

    # Identify the top-k important edges to remove (based on edge mask values)
    k = 5  # Number of edges to remove for counterfactual generation
    important_edges = explanation.edge_mask.argsort(descending=True)[:k]

    # Remove the most important edges to create a counterfactual graph
    edge_index = data.edge_index.clone()
    for i in important_edges:
        edge_index[:, i] = -1  # Mark edge for removal

    # Filter out removed edges
    edge_index = edge_index[:, edge_index[0] != -1]

    # Perturb node features based on node mask (first 4 elements only)
    node_features = data.x.clone()
    node_mask = explanation.node_mask

    # Apply perturbation to the top-k most important node features
    for node_idx in range(node_features.shape[0]):
        # Perturb the first 4 elements based on importance from node mask
        important_features = node_mask[node_idx, :4].argsort(descending=True)[:k % 4]
        for feature_idx in important_features:
            # Randomly perturb the feature value to simulate a counterfactual change
            node_features[node_idx, feature_idx] = torch.rand(1).item() * (1 - node_features[node_idx, feature_idx])

    # Create the counterfactual graph data
    counterfactual_data = data.clone()
    counterfactual_data.edge_index = edge_index
    counterfactual_data.x = node_features

    # Forward pass with modified graph to check if the prediction changes
    counterfactual_out = model(counterfactual_data.x, counterfactual_data.edge_index, counterfactual_data.batch)
    counterfactual_pred = counterfactual_out.argmax(dim=1).item()

    # Store the counterfactual results
    counterfactual = Data(x=counterfactual_data.x, edge_index=counterfactual_data.edge_index, y=counterfactual_pred)

    # Display counterfactual results
    print(f"Counterfactual Prediction: {counterfactual_pred}")
    if counterfactual_pred != original_pred:
        tot_counterfactuals += 1


        print("The prediction has changed, indicating a successful counterfactual modification.")
    else:
        print("The prediction did not change; further modifications are needed.")

    # Save natural language representations if prediction has changed
    if counterfactual_pred != original_pred:
        factual_nl_rep = build_natural_language_representation_graph_classification(data)
        counterfactual_nl_rep = build_natural_language_representation_graph_classification(counterfactual)
        
        
        factual_dict[key] = generate_factual_counterfactual_dict_aids(data, counterfactual, key)
        f_d[key] = factual_nl_rep
        cf_d[key] = counterfactual_nl_rep
        key += 1

with open('data/factual_dict.json', 'w') as f_file:
    json.dump(factual_dict, f_file, indent=4)

    
with open('data/factual_dict_description.json', 'w') as f_file:
    json.dump(f_d, f_file, indent=4)

with open('data/counterfactual_dict_description.json', 'w') as cf_file:
    json.dump(cf_d, cf_file, indent=4)
    
    
print(f"Total number of counterfactuals generated: {tot_counterfactuals/len(test_dataset)}")


