# Traffic-Prediction
Traffic-Prediction-With-GNN
About Dataset:
We validate our model on two highway traffic datasets PeMSD4 from California. The datasets are collected by the Caltrans Performance Measurement System (PeMS) in real time every 30 seconds. The traffic data are aggregated into every 5-minute interval from the raw data. The system has more than 39,000 detectors deployed on the highway in the major metropolitan areas in California. Geographic information about the sensor stations are recorded in the datasets. There are three kinds of traffic measurements considered in our experiments, including total flow, average speed, and average occupancy.

Data splitting
10181 data/target examples will be used as the training set ( 35 days )
3394 data/target examples will be used as the validation set (12 days)
3394 data/target examples will be used as the testing set (12 days)
Model layers :-
Temporal attention layer
In the temporal dimension, there exist correlations between the traffic conditions in different time slices, and the correlations are also varying under different situations. Likewise, we use an attention mechanism to adaptively attach different importance to data.



![68747470733a2f2f692e6962622e636f2f4b775843714a782f74656d702d617474656e74696f6e2e706e67](https://user-images.githubusercontent.com/113280269/191446792-5e6c2245-55a9-4bb3-a57b-3381a01f5a06.png)

It learns to attend (focus) on which part of the time segement used as input. In our case we have 12-time points So it will generate 12 by 12 weights.

Spatial attention layer
In the spatial dimension, the traffic conditions of different locations have influence among each other and the mutual influence is highly dynamic. Here, we use an attention mechanism (Feng et al. 2017) to adaptively capture the dynamic correlations between nodes in the spatial dimension.





The same as with the temporal attention; however, here the attention weights will be used inside a Graph convolution layer

Spectral graph analysis on the spatial part
Since the spatial part is represented as a graph, we will apply graph convolution to aggregate messages from neighbor nodes. The type of graph convolution that we are going to use is spectral convolution.

In spectral graph analysis, a graph is represented by its corresponding Laplacian matrix.

The properties of the graph structure can be obtained by analyzing Laplacian matrix and its eigenvalues

Laplacian matrix of a graph is defined as L = D − A,

Its normalized form is L = I − ((1/ sqrt(D) A ( 1/ sqrt(D))

where A is the adjacent matrix, I is a unit matrix, and the degree matrix D (diagnoal diagonal matrix, consisting of node degrees,at the diagonal)

The eigenvalue decomposition of the Laplacian matrix is L = UΛ(U.transpose()) , where Λ = diag([λ0, ..., λN −1]) is a diagonal matrix, and U is Fourier basis.

U is an orthogonal matrix.

The graph convolution is a convolution operation implemented by using linear operators that diagonalize in the Fourier domain to replace the classical convolution operator.

However, it is expensive to directly perform the eigenvalue decomposition on the Laplacian matrix when the scale of the graph is large. Therefore, Chebyshev polynomials are adopted to solve this problem approximately but efficiently.

Model structure -
The ASTGCN model structure
The model is composed of two ASTGCN blocks followed by a final layer Original x (input) is (32, 307, 1, 12) - Block1 > (32, 307, 64, 12) - Block2 > (32, 307, 64, 12) - permute -> (32, 12, 307,64) # -final_conv -> (32, 12, 307, 1) -reshape-> (32,307,12) -> "The target" The model is the fusion of three independent components with the same structure, which are designed to respectively model the recent, daily-periodic and weekly-periodic dependencies of the historical data. But in our case, we will only focus on the recent segment (last hour segment) i.e. X_h
![model](https://user-images.githubusercontent.com/113280269/191446563-eef74c73-7a42-42cd-88b1-4304696f2b9c.png)





Requirements:
Library used:
pytorch geometrical
Numpy
Matplotlib
Torch_geometric 2.0.4
Torch-scatter
Torch-sparse
tensorboaredx 2.5
Configuration


Link :https://ojs.aaai.org//index.php/AAAI/article/view/3881
dataset link:https://drive.google.com/drive/folders/18BdgUuKAa2BcKn90kucFEMX4zNTxKods
