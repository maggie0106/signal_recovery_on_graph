# signal_recovery_on_graph
This code is aim to reproduce the results from the paper Signal recovery on graphs: Fundamental limits of sampling strategies. [1]

The following parameters could affect the performance of this signal recovery method:
•	graph type: ring graph/random graph/random geometric graph

•	graph Fourier basis: Adjacency basis or Laplacian basis

•	node number: (N)
•	sample score: uniform sampling or experiment design sampling(experiment design sampling including: leverage score/square root leverage score)
•	sample number (m)
•	introduced noise: N~ (0, epsilon )
•	spectral decay (beta increase, the spectral decay faster)
•	signal recovery bandwidth (according to the paper this parameter should be set to max⁡(10,m^(1/(2*beta+1)))to achieve optimcal rate of convergence)





[1] Chen, Siheng, et al. "Signal recovery on graphs: Fundamental limits of sampling strategies." IEEE Transactions on Signal and Information Processing over Networks 2.4 (2016): 539-554.
