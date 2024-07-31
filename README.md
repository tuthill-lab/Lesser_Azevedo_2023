# Lesser_Azevedo_2023
Scripts and dataframes for recreating analyses in Lesser, Azevedo, et. al., 2023, published [here](https://www.nature.com/articles/s41586-024-07600-z)

Also included is a pdf of the supplemental tables "Supplemental Tables 1-3.pdf", which feature links to the reconstructed neurons.

This repository can be used with or without access to the FANC materialization: 
* Without access, dataframes are included to run through scripts for recreating analyses and plots from Lesser, Azevedo et. al., 2023.
  * For the leg matrix, the v840 matrices in the dfs_pre_to_mn folder contain the data presented in the paper, a static time point on January 14, 2024.
    * pre_to_mn_df_matched_typed_with_nt_v840.pkl is the matrix of identifiied premns, no fragments, with hemilineages and inferred neurotransmitters for local neurons
    * pre_to_mn_df_pre_match_to_pool_w_fragments_v840 is the matrix of all objects making more than 3 synapses onto leg MNs, including fragments.
* With access, scripts are included for recreating dataframes, so that the analysis pipeline can be accessed in the future as the connectome proofreading progresses.
  * For the leg connectivity matrix, run Create_t1_1_typed_left_pre_to_mn_df.ipynb to recreate the materialization v840 matrix.
  * Run Create_t1_1_typed_left_pre_to_mn_df_new.ipynb to create the current matrix, reflecting any proofreading of the segmentation.

Leg MN analyses: 
* MN_0_overview_plots.ipynb recreates data in figures 1, 2, and extended data Figure 1.
* MN_14_compare_syn_density_UMAP.ipynb compares MNs on the basis of the location of their input synapses, shown in Azevedo, Lesser, Phelps, Mark et al. 2024.
* MN_15_NBLAST_T1_MNs.ipynb compares MNs on the basis of their dendritic morphology, shown in Azevedo, Lesser, Phelps, Mark et al. 2024.
* MN_16_clustering_experiments_leg.ipynb clusters MNs on the basis of their synaptic input, analyzes the effect of moving MNs to different clusters, and recreates the shuffling experiments to create null distributions in Figure 3, Figure 4, Figure 5, and extended data Figure 4, and 5.
* MN_17_clustering_cossim_comps_leg creates the comparison of the contributions to the cosine similarity matrix for the various classes of premns, figure 3 and extended data figure 4.

For more information on the FANC (pronounced "fancy") community and how to join, see https://connectomics.hms.harvard.edu/project1
