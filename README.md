###Dependencies#########
Numpy >= 1.3, SciPy >= 0.7

1. To reproduce the AU-PR score reported in the paper run the file :prediction_nmf_with_hd.py
2. This code combines both heat diffusion and NMF matrix factorizaiton.

#####For Ranking the side effects based on the query #####
1. Run the file: top5_ranking.py
2. to run the specific query go inside the file top5_ranking.py and change the line query = "Anaemia" to other side effects. By default the code runs for Anaemia side effects.

3. The results is based on nDCG@5 rank.

#############Data#################
The data is in data folder.
1. There are 2 files in this folder
	(i) semantic_similarity_side_effects_drugs.txt of drug-drug score
		This is the edgelist of drugs and the edge weights are the semantic similarity.
		The semantic similarity is calculated using the word2vec model from:
			http://evexdb.org/pmresources/vec-space-models/wikipedia-pubmed-and-PMCw2v.bin
	
	(ii) side-effect-and-drug_name.tsv
		This is the biparite graphs between the side-effects and drugs.
		The side effects and drugs are linked by using PubChem IDs using SIDER and DRUG bank assosciations.
		More details of the data extraction:
		https://github.com/dhimmel/SIDER4/blob/master/SIDER4.ipynb

	(iii) From the file side-effect-and-drug_name.tsv we used drug_name and side_effect_name to create the bipartite graph

