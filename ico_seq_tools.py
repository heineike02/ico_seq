import sys
#Indicate operating environment and import core modules
location_input = input("what computer are you on? a = Bens, b = gpucluster, c = other   ")
location_dict = {'a': "C:\\Users\\BMH_work\\github\\expression_broad_data", 'b': "/home/heineike/github/expression_broad_data",'c':'you need to add your location to the location_dict'}
base_dir_rna_seq = location_dict[location_input]
print("rna_seq base directory is " + base_dir_rna_seq)

if sys.path[-1] != base_dir_rna_seq:
    sys.path.append(base_dir_rna_seq)
    print("Added " + base_dir_rna_seq + " to path: " )
    print(sys.path)

import os

print("I am about to import a library")
from core import expression_plots 
from core import io_library 

base_dir = os.path.normpath("C:/Users/BMH_work/Google Drive/UCSF/Yeast_colony_drop_seq")
data_processing_dir = base_dir + os.sep + "data" + os.sep
#base_dir + os.sep + os.path.normpath("expression_data") + os.sep

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar as mpl_colorbar
import matplotlib.colors as mpl_colors
#import matplotlib.colormap as cm
#from matplotlib_venn import venn2
import seaborn as sns; sns.set(style="ticks", color_codes=True)
#from sklearn import linear_model
import pickle
#import subprocess
#import networkx as nx
import scipy.stats as stats
#import statsmodels.graphics.gofplots as stats_graph
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity

# from Bio import SeqIO
# from Bio import SeqFeature as sf
# from Bio.SeqRecord import SeqRecord
# from Bio.Alphabet import generic_dna
# from Bio.Seq import Seq

# import re

from collections import Counter
# import scipy.stats as stats
# from itertools import chain
#from itertools import product
#this only works if you are online
online_input = input("are you online? Yes/No")
if online_input == "Yes": 
    import plotly.plotly as py
    import plotly.graph_objs as pygo
    import plotly.tools as pytools
    py.sign_in('heineike02_student','9dMTMZgJMgUP0YX0P5mQ')
    #py.sign_in('heineike02', 'APPjKrtARaN2ZgUYIkqr')
    
# for phylogenetic trees: 
# from ete3 import Tree

#for scraping internet data (e.g. ncbi)
#import requests
#from lxml import etree    #parses xml output

def load_process_ARO4_data(mincounts, thresh_genes, M, lowcount_thresh, scale_factor):
    #Loads and processes data in 4 steps: 
    #
    #1. Selects colonies containing at least thresh_genes with at least mincounts counts. 
    # For thresh_genes = 500, and mincounts = 1, that leaves 1642 colonies
    #2. Selects genes that have at least lowcount_thresh counts in one of the smallest M colonies (based on total number of reads)
    # For lowcount_thresh = 3 and M = 300, that leaves 1064 genes
    #
    #3. Normalizes each colony by total number of counts and multiply by the scale_factor
    #
    #4. Add a pseudocount. I use 0.1 times 1/the largest number of total counts in any remaining 
    #colony, multiplied by the scale factor.  Thus the pseudocount is 0.1 * the smallest possible value for a single read.
    #
    # scale_factor can be any float, but should be close to a typical total number of counts (ie. 10000.0) for the experiment to aid in interpretabilty. 
    # If you use 'median' then it will use the median of total counts and output that number
    #
    
    ARO4_lib_data = pd.read_table(data_processing_dir + "out_gene_exon_tagged_2000.dge.txt")
    ARO4_lib_data.set_index('GENE', inplace = True)

    #Update gene names to current SGD names
    ico_seq_synonyms = pd.read_csv(data_processing_dir + "ICO_seq_common_name_mismatches.csv")
    ico_seq_synonyms_dict = dict(zip(ico_seq_synonyms['input'],ico_seq_synonyms['symbol']))
    ARO4_lib_data.rename(index =ico_seq_synonyms_dict, inplace = True)

    #1. Select only colonies containing at least thresh_genes genes that have at least mincounts counts.  

    ngenes = []
    print('Selecting colonies containing data for at least {0:d} genes with at least {1:d} count'.format(thresh_genes, mincounts))


    cells_above_thresh = []
    for cell in ARO4_lib_data.columns:
        ngenes_gt_mincounts = len(ARO4_lib_data[cell][ARO4_lib_data[cell]>=mincounts])
        ngenes.append(ngenes_gt_mincounts)
        if ngenes_gt_mincounts >=500: 
            cells_above_thresh.append(cell)

    ARO4_lib_data_mincount = ARO4_lib_data.loc[:,cells_above_thresh]


    #2. Analyze only genes that have at least lowcount_thresh counts in one of the smallest M colonies (as determined by total number of reads)
    
    print('Selecting genes that have at least {0:d} counts in one of the smallest {1:d} colonies'.format(lowcount_thresh, M))

    #Add a row for total counts and sort descending by that row. 
    ARO4_lib_data_mincount_total = ARO4_lib_data_mincount.sum()
    ARO4_lib_data_mincount_total.name = 'total_counts'
    ARO4_lib_data_mincount = ARO4_lib_data_mincount.append(ARO4_lib_data_mincount_total)
    ARO4_lib_data_mincount.sort_values(by='total_counts', axis = 1, ascending = False, inplace=True)
    ARO4_lib_data_mincount_genes = ARO4_lib_data_mincount.drop('total_counts')

    ARO4_high_exp_genes = ARO4_lib_data_mincount_genes[ARO4_lib_data_mincount_genes.iloc[:, -M:].max(axis = 1)>=lowcount_thresh]

    #3. Normalize each colony by total counts

    print('Normalizing colonies by total counts')

    #Normalize by total counts, and multipy by median of all counts to ensure scale is close to 
    #original counts scale. 
    ARO4_high_exp_genes_norm = ARO4_high_exp_genes.copy()
    ARO4_high_exp_genes_sum = ARO4_high_exp_genes.sum()
    med = ARO4_high_exp_genes_sum.median()
    if scale_factor == 'median':
        scale_factor = med
    for col in ARO4_high_exp_genes.columns:
        ARO4_high_exp_genes_norm[col] = ARO4_high_exp_genes[col]/ARO4_high_exp_genes_sum[col]*scale_factor

    #sort genes by summed expression
    ARO4_high_exp_genes_norm['gene_totals'] = ARO4_high_exp_genes_norm.sum(axis=1)
    ARO4_high_exp_genes_norm.sort_values('gene_totals',ascending=False, inplace=True)
    ARO4_high_exp_genes_norm.drop('gene_totals',axis=1, inplace = True)

    #4. Add a pseudocount. I use 0.1 times 1/the largest number of total counts in any remaining 
    #colony, multiplied by the scale factor
    pseudocount = 0.1/max(ARO4_high_exp_genes_sum)*scale_factor    
    print('adding psuedocount of {:0.4f}'.format(pseudocount))
    ARO4_high_exp_genes_norm = ARO4_high_exp_genes_norm + pseudocount
    ARO4_high_exp_genes_norm_log10 = np.log10(ARO4_high_exp_genes_norm)

    return ARO4_high_exp_genes, ARO4_high_exp_genes_norm, ARO4_high_exp_genes_norm_log10, scale_factor
   

def average_ico_seq_data(ico_count_data, N_scale):
    # 3. Take average for all genes - sum up counts for all colonies, divide by total number of colonies
    print('Making average of all counts')
    ico_count_data_avg = ico_count_data.sum(axis = 1)/ico_count_data.shape[1]

    #4. Normalize by total counts, scale by 10K (the total for the average is about 3300 counts) 
    print('Normalizing average by total counts')

    #Normalize by total counts, and multipy by median of all counts to ensure scale is close to 
    #original counts scale. 
    ico_count_data_avg_norm = ico_count_data_avg.copy()

    ico_count_data_avg_norm = ico_count_data_avg/sum(ico_count_data_avg)*N_scale

    #5. Add a pseudocount. I use 0.1*sum(ARO4_high_exp_genes_avg)*N_scale 
    pseudocount = 0.1/sum(ico_count_data_avg)*N_scale
    print('adding psuedocount of {:0.4f}'.format(pseudocount))
    ico_count_data_avg_norm = ico_count_data_avg_norm + pseudocount
    ico_count_data_avg_norm_log10 = np.log10(ico_count_data_avg_norm)
    
    return ico_count_data_avg_norm, ico_count_data_avg_norm_log10

def process_bulk_seq_data(genes, N_scale, data_processing_dir):
#Load bulk seq data, separate out genes of interest (from high_exp_genes df), normalize and scale using N_scale

    #Loading bulk data
    #Note: changed filenames to have consistent pattern, 
    #replaced CR (\r) with LF (\n), 
    #removed all spaces (there were spaces in front of all gene names)

    bulk_exps = ['colony1', 'colony2', 'culture']
    raw_count_data = {}

    for bulk_exp in bulk_exps: 
        raw_count_data_exp = pd.read_table(data_processing_dir + bulk_exp + '_formatted_extracted_exon_count.txt', header=None,  names = ['sc_genename', bulk_exp], index_col = 0)
        raw_count_data[bulk_exp] = raw_count_data_exp


    bulk_data = pd.concat([raw_count_data[bulk_exp] for bulk_exp in bulk_exps], verify_integrity=True, axis = 1)


    #The bulk experiment data has multiple counts for many rows.  This appears to be because
    #each exon gets its own separate count, but has the same name.  
    #This makes the counts into the sum of all rows with the same name. 

    bulk_data.reset_index(inplace=True)
    grouped = bulk_data.groupby('sc_genename')
    bulk_data_exons_comb = grouped.sum()

    #Update gene names to current SGD names
    ico_seq_synonyms = pd.read_csv(data_processing_dir + os.sep + "ICO_seq_common_name_mismatches.csv")
    ico_seq_synonyms_dict = dict(zip(ico_seq_synonyms['input'],ico_seq_synonyms['symbol']))
    bulk_data_exons_comb.rename(index =ico_seq_synonyms_dict, inplace = True)

    #take out subset of genes that we are analyzing for the ICO seq dataset: 
    bulk_data_subset = bulk_data_exons_comb.loc[(set(genes) & set(bulk_data_exons_comb.index)),:]


    #3. Normalize each colony by total counts

    print('Normalizing bulk data by total counts')

    #Normalize by total counts, and multipy by median of all counts to ensure scale is close to 
    #original counts scale. 
    bulk_data_subset_norm = bulk_data_subset.copy()
    bulk_data_subset_sum = bulk_data_subset.sum()
    med = bulk_data_subset_sum.median()

    for col in bulk_data_subset.columns:
        bulk_data_subset_norm[col] = bulk_data_subset[col]/bulk_data_subset_sum[col]*N_scale


    #sort genes by summed expression
    bulk_data_subset_norm['gene_totals'] = bulk_data_subset_norm.sum(axis=1)
    bulk_data_subset_norm.sort_values('gene_totals',ascending=False, inplace=True)
    bulk_data_subset_norm.drop('gene_totals',axis=1, inplace = True)

    #4. Add a pseudocount. I use 0.1 times 1/the largest number of total counts in any remaining colony,
    # multiplied by the scaling factor N_scale. Thus the pseudocount is 0.1 * the smallest possible value for a single read.
    # The scaling factor is arbitrary, but I used to use the median number of counts in all experiments


    #pseudocount = 0.1/max(bulk_data_ARO4_subset_sum)*med   
    pseudocount = 0.1/max(bulk_data_subset_sum)*N_scale 
    print('adding psuedocount of {:0.4f}'.format(pseudocount))
    bulk_data_subset_norm = bulk_data_subset_norm + pseudocount
    bulk_data_subset_norm_log10 = np.log10(bulk_data_subset_norm)

    return bulk_data_subset_norm, bulk_data_subset_norm_log10

print("loaded ico_seq_tools.py")