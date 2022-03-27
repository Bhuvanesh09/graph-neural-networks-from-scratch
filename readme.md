Implementation of generalized graph nn architecture over pytorch with specific models like GCN, GIN et cetera.

## 3.2 GCNs

- CiteSeer dataset is downloaded using pyg :

```python
import torch_geometric as pyg
df = pyg.datasets.Planetoid("datasets/planetoid", "citeseer")
#df.data is the pandas dataframe having node_features, edge information, classes etc
```

### Difference between row, column and symmetric normalization:

Note: Experiments conducted with `num_layers=3` , `latent_dim=15`

- Row Normalization:  This scales down the neighbour’s node features before addition according to the number of neighbours of $u$. This is to ensure that the node’s own feature vector are not diluted down when a particular node has a lot of neighbours since they all would be added before they are operated on by the $W$ matrix.
- Column Normalization:  This scales down the neighbour’s (v) node features before addition according to the number of neighbours that the neighbour itself has. This method also tries to prevent dilution of certain nodes over others. This has the intent of preventing a node from dominating in it’s neighbours message aggregation just because this node has a lot of neighbours and stored larger values in its feature from its last iteration. Since it is normalizing the feature vector’s magnitudes from past iterations, its not that applicable in cases where we have less number of layers.
- Symmetric Normalization:  Symmetric normalization tries to counter both the above effect by scaling down the feature vector by the geometric mean of u’s and v’s degree, and hence its applicable for both larger number of activation layers and also lesser number of activation layer.

**Results:** Over multiple runs, we observed that the accuracy of the model ranged from ~65% to 70% validation accuracy. We failed to observe any consistent performance advantage of one particular normalization over the other to be statistically significant but in the experiments performed we saw row normalization to perform better than column normalization. But overall, symmetric normalization  was performing better on average than other two methods. We hypothesize that the difference in performance would be more apparent between the two when there are larger number of layers in play.

## 3.3 Graph Isomorphism Networks

Graph isomorphism networks have proven themselves to be worthy wherever the task is to have a good discriminatory classification between different graphs. For example: GINs performed reasonably well in contrastive learning on molecular graphs and aided in learning a good latent space representation of molecules to enumerate the chemical space ([https://arxiv.org/abs/2102.10056](https://arxiv.org/abs/2102.10056) {MolCLR})  This experiment for this assignment would help us study if such an architecture also gives advantage in node classification tasks.

$$
h_u^{(t)} = f_\theta \Bigg(  (1 + \epsilon) h_u^{(t-1)} + \sum_{v\in\mathcal{N}(u)} h_v^{(t-1)} \Bigg)
$$

The fundamental difference between GIN and GCNs which lead to difference in performance(opinion) is that $f_\theta$ in GIN is an MLP with activations between layers where as f in GCN was just an activation layer. This gives GIN better generalizability and discriminative power. 

**Result:** We observed that GINs don’t give any significant performance boost over GCNs and give accuracy of the same order of magnitude. We expect GINs to perform better when the task 

## 3.4 RNNs as Graph networks

The code is implemented in `[rnn.py](http://rnn.py)` though we couldn’t train to a good extent because of the large nature of dataset. The hack is to have a conditional COMBINE function which simply copies the neighbour as the current nodefeature and applies actual NN only at the added master node where the message of $t^{th}$ word reaches on the $t^{th}$ timestep. (Kindly see the code for a better understanding)

## 3.5 Instruction to run the code:

### Directory structure:

```bash
.
├── baseclass.py
├── datasets
│   └── planetoid
│       └── citeseer
│           ├── processed
│           │   ├── data.pt
│           │   ├── pre_filter.pt
│           │   └── pre_transform.pt
│           └── raw
│               ├── ind.citeseer.allx
│               ├── ind.citeseer.ally
│               ├── ind.citeseer.graph
│               ├── ind.citeseer.test.index
│               ├── ind.citeseer.tx
│               ├── ind.citeseer.ty
│               ├── ind.citeseer.x
│               └── ind.citeseer.y
├── gcn.py
├── gin.py
├── lightning_logs
│   └── version_637337
│       ├── checkpoints
│       └── hparams.yaml
├── main.py
├── output.txt
```

### To Install the required packages:

`pip install -r requirements.txt`

### Run the code:

`python main.py --task <gcn/gin> --layers <int> --dims <int>`