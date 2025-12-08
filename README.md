# Source code for CAT published at NDSS 2026
J. Wang, Z. Yan, J. Lan, X. Li and E. Bertino, "CAT: Can Trust be Predicted with Context-Awareness in Dynamic Heterogeneous Networks?" to appear in Proceedings of the Network and Distributed System Security (NDSS) Symposium, San Diego, CA, USA, 2026.

## Model architecture
<div align="center">
    <img src="model architecture.png" width="100%">
</div>

> CAT consists of four layers: a graph construction layer, an embedding layer, a heterogeneous attention layer, and a prediction layer. Specifically, the graph construction layer builds a contextual trust graph using streaming user-item and user-user interactions. In the embedding layer, a meta-path covering rich semantics and contextual features is first defined, and then node embeddings are initialized via Metapath2vec. Meanwhile, time information and edge attributes (e.g., ratings from users on items) are encoded to facilitate information propagation and aggregation. The heterogeneous attention layer employs a dual attention mechanism to discriminate the importance of node types and the importance of nodes within the same type. Leveraging the message-passing mechanism of GNNs, CAT achieves trust propagation with varying attentions granted on different interactions. To enhance scalability, we adopt recent-time neighbor sampling and one-hop trust propagation strategies, focusing on limited yet crucial interactions. Finally, in the prediction layer, we first generate a context embedding by averaging item embeddings within the same context. Then, two user embeddings together with the context embedding are fed into a Multi-Layer Perceptron (MLP) to predict the latent trust level between the two users within that context. To overcome the lack of context-specific trust labels, we propose a context-aware aggregator to link context-aware trust with overall trust, weighted by context importance. The integration of these layers gives CAT a comprehensive semantic understanding, which hinders potential attacks, as attackers have to consider multiple contextual and semantic factors. 

## How to start
### Step 1: Configure python environment
```shell
torch==1.13.0
torch-geometric==2.2.0
torch-scatter==2.1.0
torch-sparse==0.6.16
```

### Step 2: Run the code
```shell
cd CAT/code
python CAT_main.py
```

## Comments
This repository provides an example of how CAT runs on the Epinions dataset under clean settings. If you have any questions about the code, please feel free to ask here or contact me via email at <jwang1997@stu.xidian.edu.cn>. This work is designed based on [TGAT](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs) and [HGAT](https://github.com/BUPT-GAMMA/HGAT). Thanks for their excellent work!
