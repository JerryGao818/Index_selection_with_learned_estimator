# Automatic Index Selection with Learned Cost Estimator
The indexes selected by existing index selection model may be sub-optimal due to the usuage of the cost estimation model in optimizer. To address this problem, we present an automatic index selector with GCN based cost estimator to improve index selection quality. In order to get accurate estimations of an index's benefit, we design a deep learning based cost estimator to predict a query's execution time on certain indexes. In particular, we regard query plans as graphs and develop a graph convolutional network (GCN) based model to learn features from queries and indexes. After that, we design a reinforcement learning based index selector considering the relationships among indexes to make index recommendation, and combine our cost estimator to evaluate the benefit of each index in the index selection model. We offer the source code and the data collected by us in JOB workload.


****

## Get Started
The project works on PostgreSQL 11. You also need to install [HypoPG](https://hypopg.readthedocs.io/en/latest/) due to the usage of what-if interface.

If you want to collect your own data, run GenerateTrainingData.py in PostgreGenerator. Or you can use the data collected by us in [IMDBData](https://drive.google.com/file/d/1uRBtv-pnMflxpeeMrD_nAa8E_IR_fsTv/view?usp=sharing) to test the performance of the model. Oringinal data can be found in (https://www.imdb.com/interfaces/).

To train the cost estimation model, unzip IMDBData and rename it as data in CostEstimator and run CostEstimator.py.

To train the index selection model and select index, input the database name and account in configure.ini and run EntryM3DP.py in Entry. You should train the cost estimation model first.

## Acknowledgements
The cost estimation part is based on "Yuan H , Li G , Feng L , et al. Automatic View Generation with Deep Learning and Reinforcement Learning[C]// 2020 IEEE 36th International Conference on Data Engineering (ICDE). IEEE, 2020."

The index selecion part is based on "Lan H, Bao Z, Peng Y. An Index Advisor Using Deep Reinforcement Learning[C]//Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 2020: 2105-2108."  [IndexAdvisor](https://github.com/rmitbggroup/IndexAdvisor).



