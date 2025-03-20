import numpy as np
import scanpy as sc
import anndata

import time
import os, wget

import cupy as cp
import cudf

from cuml.decomposition import PCA
from cuml.manifold import TSNE
from cuml.cluster import KMeans
from cuml.preprocessing import StandardScaler

import rapids_scanpy_funcs
import utils

import warnings
warnings.filterwarnings('ignore', 'Expected ')
warnings.simplefilter('ignore')
import pandas as pd
from sh import gunzip
import scipy
from scipy import sparse

def load_parameters():
    D_R_mtx=pd.read_csv("/data/drug_receptor_mtx.csv",index_col=0)
    GPCR_type_df=pd.read_csv("/data/GPCR_df.csv",index_col=0)

    drug_list=D_R_mtx.index.to_list()
    GPCR_list=["HTR1A","HTR1B","HTR1D","HTR1E","HTR2A","HTR2B","HTR2C",
    "HTR3A","HTR4","HTR5A","HTR6","HTR7","DRD1","DRD2","DRD3","DRD4","DRD5",
    "HRH1","HRH2","HRH3","CHRM1","CHRM2","CHRM3","CHRM4","CHRM5",
    "ADRA1A","ADRA1B","ADRA2A","ADRA2B","ADRA2C","ADRB1","ADRB2"]
    D_R_mtx.columns=GPCR_list
    return D_R_mtx,GPCR_type_df,drug_list,GPCR_list

def set_parameters_for_preprocess(GPCR_list):
    params = {}  # Create an empty dictionary to store parameters
    # maximum number of cells to load from files
    params["USE_FIRST_N_CELLS"] = 300000
    
    # Set MITO_GENE_PREFIX
    params['MITO_GENE_PREFIX'] = "mt-"
    
    # Set markers
    markers = ["CX3CR1","CLDN5","GLUL","NDRG2","PCDH15","PLP1","MBP","SATB2","SLC17A7",
               "SLC17A6","GAD2","GAD1","SNAP25"]
    markers.extend(GPCR_list)
    params['markers'] = [str.upper() for str in markers]
    
    # Set cell filtering parameters
    params['min_genes_per_cell'] = 200
    params['max_genes_per_cell'] = 6000
    
    # Set gene filtering parameters
    params['min_cells_per_gene'] = 1
    params['n_top_genes'] = 4000
    
    # Set PCA parameters
    params['n_components'] = 50
    
    # Set Batched PCA parameters
    params['pca_train_ratio'] = 0.35
    params['n_pca_batches'] = 10
    
    # Set t-SNE parameters
    params['tsne_n_pcs'] = 20
    
    # Set k-means parameters
    params['k'] = 35
    
    # Set KNN parameters
    params['n_neighbors'] = 15
    params['knn_n_pcs'] = 50
    
    # Set UMAP parameters
    params['umap_min_dist'] = 0.3
    params['umap_spread'] = 1.0
    
    return params


def preprocess_adata_in_bulk(adata_path,label=None,add_markers=None):
    preprocess_start = time.time()
    D_R_mtx,GPCR_type_df,drug_list,GPCR_list=load_parameters()
    # Set parameters
    params = set_parameters_for_preprocess(GPCR_list)

    # Add any additional markers if provided
    if add_markers is not None:
        # Ensure the additional markers are in uppercase for consistency
        add_markers = [marker.upper() for marker in add_markers]
        # Append the additional markers to the markers list in the parameters
        params['markers'].extend(add_markers)
    
    #preprocess in bulk
    print("preprocess_in_bulk")
    adata = anndata.read_h5ad(adata_path)
    if label !=None:
        adata=adata[adata.obs["label"]==label]
    genes = cudf.Series(adata.var_names).str.upper()
    barcodes = cudf.Series(adata.obs_names)
    is_label=False
    # Initialize labels dataframe if "label" column exists in adata.obs
    if "label" in adata.obs.columns:
        is_label=True
        original_labels = adata.obs["label"].copy()
    #if len(adata.obs["label"])>0:
    #    is_label=True
    #    labels=cudf.DataFrame(adata.obs["label"])
    #    labels = cudf.DataFrame({"barcode": barcodes.reset_index(drop=True), "label": adata.obs["label"]})
        #labels= cudf.DataFrame(adata.obs['label'])
    sparse_gpu_array = cp.sparse.csr_matrix(adata.X)
    sparse_gpu_array,filtered_barcodes = rapids_scanpy_funcs.filter_cells(sparse_gpu_array, min_genes=params['min_genes_per_cell'],
                                                        max_genes=params['max_genes_per_cell'],barcodes=barcodes)
    sparse_gpu_array, genes = rapids_scanpy_funcs.filter_genes(sparse_gpu_array, genes, 
                                                            min_cells=params['min_cells_per_gene'])
    """sparse_gpu_array, genes, marker_genes_raw = \
    rapids_scanpy_funcs.preprocess_in_batches(adata_path, 
                                              params['markers'], 
                                              min_genes_per_cell=params['min_genes_per_cell'], 
                                              max_genes_per_cell=params['max_genes_per_cell'], 
                                              min_cells_per_gene=params['min_cells_per_gene'], 
                                              target_sum=1e4, 
                                              n_top_genes=params['n_top_genes'],
                                              max_cells=params["USE_FIRST_N_CELLS"])
    """
    markers=params['markers'].copy()
    df=genes.to_pandas()
    
    # Before loop: create a set of markers to remove
    markers_to_remove = set()

    # Inside the loop, just add to the set if the marker needs to be removed
    for marker in markers:
        if not marker in df.values:
            print(f"{marker} is not included")
            markers_to_remove.add(marker)
            print(f"{marker} is removed from marker list")

    # After loop: remove the markers that are not found
    for marker in markers_to_remove:
        markers.remove(marker)
    
    print(markers)            
    tmp_norm = sparse_gpu_array.tocsc()
    marker_genes_raw = {
        ("%s_raw" % marker): tmp_norm[:, genes[genes == marker].index[0]].todense().ravel()
        for marker in markers
    }

    del tmp_norm
    ## Regress out confounding factors (number of counts, mitochondrial gene expression)
    # calculate the total counts and the percentage of mitochondrial counts for each cell
    mito_genes = genes.str.startswith(params['MITO_GENE_PREFIX'])
    n_counts = sparse_gpu_array.sum(axis=1)
    percent_mito = (sparse_gpu_array[:,mito_genes].sum(axis=1) / n_counts).ravel()
    n_counts = cp.array(n_counts).ravel()
    percent_mito = cp.array(percent_mito).ravel()
    
    # regression
    print("perform regression")
    
    sparse_gpu_array = rapids_scanpy_funcs.regress_out(sparse_gpu_array.tocsc(), n_counts, percent_mito)
    del n_counts, percent_mito, mito_genes
    
    
    # scale
    print("perform scale")
    mean = sparse_gpu_array.mean(axis=0)
    sparse_gpu_array -= mean
    stddev = cp.sqrt(sparse_gpu_array.var(axis=0))
    sparse_gpu_array /= stddev
    sparse_gpu_array = sparse_gpu_array.clip(a_max=10)
    del mean, stddev
    
    preprocess_time = time.time()
    print("Total Preprocessing time: %s" % (preprocess_time-preprocess_start))
    
    ## Cluster and visualize
    adata = anndata.AnnData(sparse_gpu_array.get())
    adata.var_names = genes.to_pandas()
    adata.obs_names = filtered_barcodes.to_pandas()
    print(f"shape of adata: {adata.X.shape}")
    
    # Restore labels after preprocessing
    if is_label:
        # Convert filtered_barcodes to a pandas Series
        filtered_barcodes_host = filtered_barcodes.to_pandas()  # <- 追加: データをホストに移動
        filtered_labels = original_labels.loc[filtered_barcodes_host].values
        adata.obs["label"] = filtered_labels
    
    del sparse_gpu_array, genes
    print(f"shape of adata: {adata.X.shape}")
    
    GPCR_df=pd.DataFrame()
    for name, data in marker_genes_raw.items():
        adata.obs[name] = data.get()
        if   name[:-4] in GPCR_list:
            GPCR_df[name]=data.get()
        
    # Deminsionality reduction
    #We use PCA to reduce the dimensionality of the matrix to its top 50 principal components.
    #If the number of cells was smaller, we would use the command 
    # `adata.obsm["X_pca"] = cuml.decomposition.PCA(n_components=n_components, output_type="numpy").fit_transform(adata.X)` 
    # to perform PCA on all the cells.
    #However, we cannot perform PCA on the complete dataset using a single GPU. 
    # Therefore, we use the batched PCA function in `utils.py`, which uses only a fraction 
    # of the total cells to train PCA.
    adata = utils.pca(adata, n_components=params["n_components"], 
                  train_ratio=params["pca_train_ratio"], 
                  n_batches=params["n_pca_batches"],
                  gpu=True)
    
    #t-sne + k-means
    adata=tsne_kmeans(adata,params['tsne_n_pcs'],params['k'])
    
    #UMAP + Graph clustering
    adata=UMAP_adata(adata,params["n_neighbors"],params["knn_n_pcs"],
                     params["umap_min_dist"],params["umap_spread"])
   
    #calculate response to antipsychotics
    adata=calc_drug_response(adata,GPCR_df,GPCR_type_df,drug_list,D_R_mtx)
    
    #calculate clz selectivity
    adata=calc_clz_selective_cell(adata,drug_list)
    
    #save preprocessed adata 
    file_root, file_extension = os.path.splitext(adata_path)
    # Append '_processed' to the root and add the extension back
    processed_file_path = f"{file_root}_processed{file_extension}"
    adata.write(processed_file_path)
    
    return adata

def tsne_kmeans(adata,tsne_n_pcs,k):
    adata.obsm['X_tsne'] = TSNE().fit_transform(adata.obsm["X_pca"][:,:tsne_n_pcs])
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0).fit(adata.obsm['X_pca'])
    adata.obs['kmeans'] = kmeans.labels_.astype(str) 
    print("t-sne + k-means")       
    sc.pl.tsne(adata, color=["kmeans"])
    return adata

def UMAP_adata(adata,n_neighbors,knn_n_pcs,umap_min_dist,umap_spread):
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=knn_n_pcs,
                    method='rapids')
    sc.tl.umap(adata, min_dist=umap_min_dist, spread=umap_spread,
               method='rapids')
    sc.tl.louvain(adata, flavor='rapids')
    print("UMAP louvain")
    sc.pl.umap(adata, color=["louvain"])
    adata.obs['leiden'] = rapids_scanpy_funcs.leiden(adata)
    print("UMAP leiden")
    sc.pl.umap(adata, color=["leiden"])
    return adata

def preprocess_adata_in_batch(adata_path,max_cells):
    preprocess_start = time.time()
    D_R_mtx,GPCR_type_df,drug_list,GPCR_list=load_parameters()
    #set parameters
    params=set_parameters_for_preprocess(GPCR_list)
    
    #preprocess in batch
    print("preprocess_in_batches")
    sparse_gpu_array, genes, marker_genes_raw = \
    rapids_scanpy_funcs.preprocess_in_batches(adata_path, 
                                              params['markers'], 
                                              min_genes_per_cell=params['min_genes_per_cell'], 
                                              max_genes_per_cell=params['max_genes_per_cell'], 
                                              min_cells_per_gene=params['min_cells_per_gene'], 
                                              target_sum=1e4, 
                                              n_top_genes=params['n_top_genes'],
                                              max_cells=max_cells)#params["USE_FIRST_N_CELLS"]
    
    print("marker_genes_raw")
    print(marker_genes_raw)
    ## Regress out confounding factors (number of counts, mitochondrial gene expression)
    # calculate the total counts and the percentage of mitochondrial counts for each cell
    mito_genes = genes.str.startswith(params['MITO_GENE_PREFIX'])

    n_counts = sparse_gpu_array.sum(axis=1)
    percent_mito = (sparse_gpu_array[:,mito_genes].sum(axis=1) / n_counts).ravel()

    n_counts = cp.array(n_counts).ravel()
    percent_mito = cp.array(percent_mito).ravel()
    
    # regression
    print("perform regression")
    sparse_gpu_array = rapids_scanpy_funcs.regress_out(sparse_gpu_array.tocsc(), n_counts, percent_mito)
    del n_counts, percent_mito, mito_genes
    
    # scale
    print("perform scale")
    mean = sparse_gpu_array.mean(axis=0)
    sparse_gpu_array -= mean
    stddev = cp.sqrt(sparse_gpu_array.var(axis=0))
    sparse_gpu_array /= stddev
    sparse_gpu_array = sparse_gpu_array.clip(a_max=10)
    del mean, stddev
    
    preprocess_time = time.time()
    print("Total Preprocessing time: %s" % (preprocess_time-preprocess_start))
    
    ## Cluster and visualize
    adata = anndata.AnnData(sparse_gpu_array.get())
    adata.var_names = genes.to_pandas()
    del sparse_gpu_array, genes
    print(f"shape of adata: {adata.X.shape}")
    
    GPCR_df=pd.DataFrame()
    for name, data in marker_genes_raw.items():
        print(len(adata.obs[name]))
        print(len(data.get()))
        adata.obs[name] = data.get()
        if   name[:-4] in GPCR_list:
            GPCR_df[name]=data.get()
        
    # Deminsionality reduction
    #We use PCA to reduce the dimensionality of the matrix to its top 50 principal components.
    #If the number of cells was smaller, we would use the command 
    # `adata.obsm["X_pca"] = cuml.decomposition.PCA(n_components=n_components, output_type="numpy").fit_transform(adata.X)` 
    # to perform PCA on all the cells.
    #However, we cannot perform PCA on the complete dataset using a single GPU. 
    # Therefore, we use the batched PCA function in `utils.py`, which uses only a fraction 
    # of the total cells to train PCA.
    adata = utils.pca(adata, n_components=params["n_components"], 
                  train_ratio=params["pca_train_ratio"], 
                  n_batches=params["n_pca_batches"],
                  gpu=True)
    
    #t-sne + k-means
    adata.obsm['X_tsne'] = TSNE().fit_transform(adata.obsm["X_pca"][:,:params['tsne_n_pcs']])
    kmeans = KMeans(n_clusters=params['k'], init="k-means++", random_state=0).fit(adata.obsm['X_pca'])
    adata.obs['kmeans'] = kmeans.labels_.astype(str)        
    sc.pl.tsne(adata, color=["kmeans"])
    #sc.pl.tsne(adata, color=["SNAP25_raw"], color_map="Blues", vmax=1, vmin=-0.05)
    
    #UMAP + Graph clustering
    sc.pp.neighbors(adata, n_neighbors=params["n_neighbors"], n_pcs=params["knn_n_pcs"],
                    method='rapids')
    sc.tl.umap(adata, min_dist=params["umap_min_dist"], spread=params["umap_spread"],
               method='rapids')
    sc.tl.louvain(adata, flavor='rapids')
    sc.pl.umap(adata, color=["louvain"])
    adata.obs['leiden'] = rapids_scanpy_funcs.leiden(adata)
    sc.pl.umap(adata, color=["leiden"])
    #sc.pl.umap(adata, color=["SNAP25_raw"], color_map="Blues", vmax=1, vmin=-0.05)
    
    #calculate response to antipsychotics
    #noramlize GPCR expression levels
    GPCR_adata=anndata.AnnData(X=GPCR_df)
    GPCR_adata_norm=sc.pp.normalize_total(GPCR_adata,target_sum=1e4,inplace=False)['X']
    GPCR_adata_norm_df=pd.DataFrame(GPCR_adata_norm,columns=GPCR_adata.var.index)
    norm_df=pd.DataFrame(GPCR_adata_norm)
    norm_col=[str[:-4] for str in GPCR_df.columns]
    norm_df.columns=norm_col
    
    GPCR_type_df=GPCR_type_df[GPCR_type_df.receptor_name.isin(norm_col)]
    
    Gs=GPCR_type_df[GPCR_type_df.type=="Gs"]["receptor_name"].values
    Gi=GPCR_type_df[GPCR_type_df.type=="Gi"]["receptor_name"].values
    Gq=GPCR_type_df[GPCR_type_df.type=="Gq"]["receptor_name"].values
    
    cAMP_df=pd.DataFrame(columns=drug_list)
    Ca_df=pd.DataFrame(columns=drug_list)
    for drug in drug_list:
        Gs_effect=(norm_df.loc[:,Gs]/D_R_mtx.loc[drug,Gs]).sum(axis=1) #TODO ki値で割り算するときにlog換算すべきか
        Gi_effect=(norm_df.loc[:,Gi]/D_R_mtx.loc[drug,Gi]).sum(axis=1)
        Gq_effect=(norm_df.loc[:,Gq]/D_R_mtx.loc[drug,Gq]).sum(axis=1)
        cAMPmod=Gi_effect-Gs_effect #Giの阻害→cAMP上昇、Gsの阻害→cAMP低下
        Camod=-Gq_effect #Gq阻害→Ca低下
        cAMP_df[drug]=cAMPmod
        Ca_df[drug]=Camod
        
    cAMP_df.index=adata.obs_names
    Ca_df.index=adata.obs_names
    Ca_df=Ca_df+10**(-4)
    for drug in drug_list:
        adata.obs['cAMP_%s'%drug]=cAMP_df[drug]
        adata.obs['Ca_%s'%drug]=Ca_df[drug]
    
    #save preprocessed adata 
    file_root, file_extension = os.path.splitext(adata_path)
    # Append '_processed' to the root and add the extension back
    processed_file_path = f"{file_root}_processed{file_extension}"
    adata.write(processed_file_path)
    
    return adata

def calc_drug_response(adata,GPCR_df,GPCR_type_df,drug_list,D_R_mtx):
    #noramlize GPCR expression levels
    GPCR_adata=anndata.AnnData(X=GPCR_df)
    GPCR_adata_norm=sc.pp.normalize_total(GPCR_adata,target_sum=1e4,inplace=False)['X']
    GPCR_adata_norm_df=pd.DataFrame(GPCR_adata_norm,columns=GPCR_adata.var.index)
    norm_df=pd.DataFrame(GPCR_adata_norm)
    norm_col=[str[:-4] for str in GPCR_df.columns]
    norm_df.columns=norm_col
    
    GPCR_type_df=GPCR_type_df[GPCR_type_df.receptor_name.isin(norm_col)]
    
    Gs=GPCR_type_df[GPCR_type_df.type=="Gs"]["receptor_name"].values
    Gi=GPCR_type_df[GPCR_type_df.type=="Gi"]["receptor_name"].values
    Gq=GPCR_type_df[GPCR_type_df.type=="Gq"]["receptor_name"].values
    
    cAMP_df=pd.DataFrame(columns=drug_list)
    Ca_df=pd.DataFrame(columns=drug_list)
    for drug in drug_list:
        Gs_effect=(norm_df.loc[:,Gs]/D_R_mtx.loc[drug,Gs]).sum(axis=1) #TODO ki値で割り算するときにlog換算すべきか
        Gi_effect=(norm_df.loc[:,Gi]/D_R_mtx.loc[drug,Gi]).sum(axis=1)
        Gq_effect=(norm_df.loc[:,Gq]/D_R_mtx.loc[drug,Gq]).sum(axis=1)
        cAMPmod=Gi_effect-Gs_effect #Giの阻害→cAMP上昇、Gsの阻害→cAMP低下
        Camod=-Gq_effect #Gq阻害→Ca低下
        cAMP_df[drug]=cAMPmod
        Ca_df[drug]=Camod
        
    cAMP_df.index=adata.obs_names
    Ca_df.index=adata.obs_names
    Ca_df=Ca_df+10**(-4)
    for drug in drug_list:
        adata.obs['cAMP_%s'%drug]=cAMP_df[drug]
        adata.obs['Ca_%s'%drug]=Ca_df[drug]
        
    return adata

def calc_clz_selective_cell(adata,drug_list):
    selectivity_threshold=1.5
    adata.obs["is_clz_activated"]=np.zeros(len(adata.obs))
    adata.obs["is_clz_activated"][adata.obs["cAMP_CLOZAPINE"]>10]=1
    adata.obs["is_clz_activated"]=adata.obs["is_clz_activated"].astype("category")
    
    adata.obs["is_clz_inhibited"]=np.zeros(len(adata.obs))
    adata.obs["is_clz_inhibited"][adata.obs["cAMP_CLOZAPINE"]<-10]=1
    adata.obs["is_clz_inhibited"]=adata.obs["is_clz_inhibited"].astype("category")
    
    drug_list_temp=drug_list.copy()
    drug_list_temp.remove("CLOZAPINE")

    for idx,drug in enumerate(drug_list_temp):
        #print(idx)
        if idx==0:
            cAMP_mean=adata.obs["cAMP_%s"%drug]
        else:
            cAMP_mean=cAMP_mean+adata.obs["cAMP_%s"%drug]
        #print(adata.obs["cAMP_%s"%drug])
    cAMP_mean=cAMP_mean/len(drug_list_temp)
    adata.obs["cAMP_mean_other_than_czp"]=cAMP_mean
    adata.obs["cAMP_clz_selectivity"]=adata.obs["cAMP_CLOZAPINE"]**2/cAMP_mean**2
    
    adata.obs["is_clz_selective"]=np.zeros(len(adata.obs))
    adata.obs["is_clz_selective"][(adata.obs["cAMP_clz_selectivity"]>selectivity_threshold)&(adata.obs["cAMP_CLOZAPINE"]>0)]=1
    adata.obs["is_clz_selective"]=adata.obs["is_clz_selective"].astype("category")
    
    print("clz selective cells")
    sc.pl.umap(adata, color=["is_clz_selective"])
    
    print("calculating gene marker of clz selective cell")
    sc.tl.rank_genes_groups(adata, 'is_clz_selective', method='wilcoxon')
    #sc.pl.rank_genes_groups(adata, n_genes=30, sharey=False)

    return(adata)

def create_GPCR_pattern(n_pattern):
    D_R_mtx,GPCR_type_df,drug_list,GPCR_list=load_parameters()
    # 重複を避けるために使用するセット
    unique_patterns_set = set()

    # 結果を保存するための辞書
    pattern_dict = {}

    # 1万種類の独自の活性化パターンを生成
    i = 0
    while len(unique_patterns_set) < n_pattern:
        # ランダムな活性化パターンを生成（0はFalse、1はTrueとする）
        random_pattern = np.random.randint(2, size=len(GPCR_list2))
        # パターンを文字列に変換してハッシュ可能にする
        pattern_str = ''.join(map(str, random_pattern))

        # このパターンがまだ見つかっていない場合は保存
        if pattern_str not in unique_patterns_set:
            unique_patterns_set.add(pattern_str)
            pattern_dict[f"Pattern_{i+1}"] = {gpcr: bool(val) for gpcr, val in zip(GPCR_list2, random_pattern)}
            i += 1
            
    # pattern_dictをデータフレームに変換
    pattern_df = pd.DataFrame.from_dict(pattern_dict, orient='index').reset_index(drop=True)
    return pattern_df