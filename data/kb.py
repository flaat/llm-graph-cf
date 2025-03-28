cora_description = """
The Cora dataset is a widely used benchmark for graph-based learning tasks, specifically node classification in citation networks. It consists of 2,708 scientific publications classified into seven distinct research topics. The dataset is represented as a graph where each node corresponds to a paper, and edges represent citation links between papers. If a paper cites or is cited by another paper, an undirected edge is created between the two corresponding nodes.
Features:
Each node is described by a 1,433-dimensional binary feature vector, where each dimension indicates the presence or absence of a specific word from the paper’s abstract. This feature representation helps capture the content of each publication.In the current format the nodes features are represented by a list of the words in the word vector that have value 1.
Classes:
The nodes are categorized into seven classes, which correspond to the research topic of each paper:
1. Case_Based: Papers focused on case-based learning and case-based reasoning methods.
2. Genetic_Algorithms: Research related to optimization and search algorithms inspired by natural selection and genetics.
3. Neural_Networks: Publications on neural network architectures, learning algorithms, and their applications.
4. Probabilistic_Methods: Papers covering probabilistic models, including Bayesian networks and probabilistic graphical models.
5. Reinforcement_Learning: Research discussing methods for training agents to make sequential decisions in dynamic environments.
6. Rule_Learning: Publications on rule-based learning methods, including decision rules and logic programming.
7. Theory: Theoretical studies, including mathematical foundations of machine learning and algorithmic theory.
Graph Structure:
Nodes: 2,708 papers, each labeled with one of the seven classes.
Edges: 5,429 citation links between papers, forming a sparse and undirected graph structure."""


citeseer_description = """
The CiteSeer dataset is a citation network composed of scientific publications. It's widely used in machine learning and network analysis, especially for tasks involving graph neural networks (GNNs).
Key Features:
- Nodes: 3,312 documents (scientific publications).
- Edges: 4,732 citation links. Each edge represents a citation from one document to another. The graph is undirected, meaning that if document A cites document B, there is a link between both.
- Node Features: Each document is described by a 3,703-dimensional binary feature vector. Each dimension corresponds to a unique word from the dictionary, and the value is 1 if the word appears in the document and 0 otherwise (bag-of-words model).
- Classes: Each document belongs to one of six classes:
  1. Agents
  2. Artificial Intelligence
  3. Database
  4. Information Retrieval
  5. Machine Learning
  6. Human-Computer Interaction
Usage in Machine Learning:
- Node Classification: Predict the category of a document based on its content and citation relationships.
- Graph Representation Learning: Learn embeddings that capture both textual content and citation network structure.
- Benchmarking Graph Neural Networks: Evaluate GNN architectures like GCN, GAT, and others.
"""


aids_dataset_description = """ The AIDS dataset is a graph classification benchmark widely used to evaluate graph neural networks (GNNs) in tasks related to molecular property prediction. It provides a structured representation of molecular compounds, where each molecule is encoded as a graph. In these graphs, nodes represent atoms, edges represent chemical bonds, and each graph corresponds to a molecule labeled as either active (a) or inactive (i) against HIV.

Node Features and Labels
Nodes in the dataset correspond to individual atoms in the molecule and carry both labels and attributes that describe their chemical properties. The node labels, derived from the atom's chemical symbol, are mapped to integer values. For instance, carbon (C) is labeled as 0, oxygen (O) as 1, nitrogen (N) as 2, and so on, covering a wide variety of atom types found in the dataset. In addition to their labels, nodes include attributes such as:

chem: A feature that encodes specific chemical properties of the atom.
charge: Represents the electrical charge of the atom.
x, y: Spatial coordinates that define the atom's position within the molecular structure.
This rich feature set allows for nuanced representations of molecular structures, making the dataset ideal for testing the ability of GNNs to capture both topological and chemical information.

Edge Features and Labels
Edges in the graph represent the chemical bonds between atoms. Each bond is labeled based on its valence, which indicates the type of bond: single, double, or triple. These bond types are mapped to integer values (0 for single bonds, 1 for double bonds, and 2 for triple bonds). This encoding ensures that the dataset captures not only the connectivity of the molecules but also the nature of the relationships between atoms."""


        
def get_system_prompt(perturbation_type, task):
  
  
  
  
    dict_node_classification = """ infos={Target_Node_ID:'', Factual_Target_Node_Neighbors: [], Factual_Target_Node_Features: [], Counterfactual_Target_Node_Neighbors: [],
    Counterfactual_Target_Node_Features: [], Factual_Target_Node_Class: '', Counterfactual_Target_Node_class: '', Target_Node_Features_Changed_From_Factual_To_Counterfactual: [], Natural_Language_Explanation: ''}"""

    dict_graph_classification = """ infos= {
        "Factual_Target_Graph_Class": None  ,
        "CounterFactual_Target_Graph_Class": None ,
        "Node_Differences": [],
        "Edge_Differences":  [] ,
        "Natural_Language_Explanation": None
    }"""
          
    dict_g = dict_node_classification if task == "node" else dict_graph_classification
  
  
    return "You are a language model that generates responses in a coherent and well-structured discourse style. "\
      "Avoid using bullet points, lists, or numerical outlines. Provide your responses in complete sentences and paragraphs, "\
      "explaining concepts clearly and concisely in a continuous flow."\
      f"You must take a factual example , a graph, and its counterfactual example, another graph, generated via {perturbation_type} perturbation and you must explain the differences."\
      "Counterfactual Explanation: A counterfactual explanation refers to a type of explanation in machine learning and artificial intelligence that describes how altering certain input features can change the output of a model. It answers 'what if' scenarios by identifying minimal changes necessary to achieve a different desired outcome. Counterfactual explanations provide insights into the decision-making process of complex models, enhancing transparency and interpretability."\
      "For example, consider a credit scoring model that denies a loan application. A counterfactual explanation might be: 'If your annual income had been $50,000 instead of $45,000, your loan would have been approved.' This helps the applicant understand what specific change could lead to a different decision."\
      "Counterfactual Explanations in Graphs: Graphs are data structures consisting of nodes (vertices) and edges that represent relationships between entities. Graphs are prevalent in various domains such as social networks, biological networks, and knowledge graphs. When machine learning models, particularly Graph Neural Networks (GNNs), are applied to graph data, interpreting their predictions becomes challenging due to the complexity of the graph structure."\
      "Counterfactual explanations in graphs aim to provide insights into how changes in the graph's structure or node features can alter the model's predictions. This involves identifying minimal modifications to the graph that would change the outcome for a specific node or the entire graph."\
      f"You MUST fill this dictionary taking into account that Node_Differences must contain JUST the nodes' id where the change happen, the Edge_Differences must contain a list of tuple representing the nodes involved in the edges: {dict_g}"