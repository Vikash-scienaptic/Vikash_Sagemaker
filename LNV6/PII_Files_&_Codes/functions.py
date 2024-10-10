import pandas as pd
import numpy as np

class my_funcs():
    def read_and_save(read_path,save_path):
        data = pd.read_csv(read_path)
        data.to_csv('~'+save_path)
        return 'File saved at: ' + save_path

    def search_col(data,sub_str):
        return print([i for i in data.columns.to_list() if sub_str.lower() in i.lower()])


# Given by Ashwin G
def rnkOrderingN(data, bad_col, wgt_col, score_col, n_bkts=10,asc=False):
    if wgt_col is None:
        wgt_col = 'wt'
        data[wgt_col] = 1
    data = data.sort_values(by=[score_col]).reset_index()
    data["cum_wgt"] = data[wgt_col].cumsum()
    total = data[wgt_col].sum()
    flag = 1
#     data.reset_index(inplace=True)
    data["bucket"]=flag
    b = 1/n_bkts
    for bin_lvl in np.arange(b,1,b):
        data["bucket"] = np.where(data["cum_wgt"]>=total*bin_lvl,flag+1,data["bucket"])
        flag+=1
    grouped = data.groupby('bucket', as_index=True)
    data["labels"] = data[bad_col] * data[wgt_col]
    agg1 = grouped[score_col].min()
    agg2 = grouped[score_col].max()
    agg3 = grouped[wgt_col].sum()
    agg4 = grouped["labels"].sum()
    agg5 = grouped[score_col].mean()
    ret_tbl = pd.DataFrame({"min_score": agg1, "max_score": agg2, "total": agg3, "Bads": agg4, "mean_score": agg5})
    ret_tbl["Goods"] = ret_tbl["total"] - ret_tbl["Bads"]
    ret_tbl["Bad_Rate"] = np.round(ret_tbl["Bads"] / ret_tbl["total"], 4) * 100
    ret_tbl = ret_tbl.sort_index(axis=0, ascending=asc)
    ret_tbl.index = range(n_bkts)
    ret_tbl["Cum_Total"] = round((ret_tbl.total / ret_tbl.total.sum()).cumsum()*100,2)
    ret_tbl["Cum_Bad"] = np.round((ret_tbl.Bads / ret_tbl.Bads.sum()).cumsum(), 4) * 100
    ret_tbl["Cum_Good"] = np.round((ret_tbl.Goods / ret_tbl.Goods.sum()).cumsum(), 4) * 100
    ret_tbl['KS'] = np.abs(ret_tbl["Cum_Bad"] - ret_tbl["Cum_Good"])
    ret_tbl['Bad_Rate_Diff'] = np.abs(ret_tbl["Bad_Rate"] - (ret_tbl["mean_score"] *100))
    gini = ret_tbl["Cum_Bad"][0] * ret_tbl["Cum_Good"][0] / (2 * 100 * 100)
    for i in range(1, n_bkts):
        gini += (ret_tbl["Cum_Bad"][i] + ret_tbl["Cum_Bad"][i - 1]) * (ret_tbl["Cum_Good"][i] -
                                                                       ret_tbl["Cum_Good"][i - 1]) / (2 * 100 * 100)
    gini = np.round(2. * gini - 1., 4) * 100
    ks = np.round(ret_tbl.KS.max(), 2)
    coltitles = ["min_score", "max_score", "mean_score", "total", "Goods", "Bads", "Bad_Rate", "Cum_Total","Cum_Good","Cum_Bad", "KS","Bad_Rate_Diff"]
#     print({"Gini": gini, "KS": ks})
    ret_tbl.index +=1
    return {"Gini": gini, "KS": ks},ret_tbl[coltitles]



# Given by Barsath

def rnkOrdering2(data_org, bad_col, wgt_col, score_col, n_bkts):
    if wgt_col is None:
        wgt_col = 'wt'
        #data_org[wgt_col] = 1
    #data_org[score_col] = -1 * data_org[score_col]
    data = data_org.sort_values(by=[score_col]).reset_index()
    data["cum_wgt"] = data[wgt_col].cumsum()
    total = data[wgt_col].sum()
    bin_lvl = 0.2
    flag = 1
    data.reset_index(inplace=True)
    for i in range(len(data)):
        if (data.loc[i, "cum_wgt"] >= total * bin_lvl) and (bin_lvl < 0.9):
            bin_lvl = bin_lvl + 0.2
            flag = flag + 1
            data.loc[i, "bucket"] = flag
        else:
            data.loc[i, "bucket"] = flag
#     data['bucket'] = pd.qcut(data[score_col], n_bkts)
    grouped = data.groupby('bucket', as_index=True)
    data["labels"] = data[bad_col] * data[wgt_col]
    agg1 = grouped[score_col].min()
    agg2 = grouped[score_col].max()
    agg3 = grouped[wgt_col].sum()
    agg4 = grouped["labels"].sum()
    agg5 = np.round(grouped[score_col].mean() * 100, 2)
    ret_tbl = pd.DataFrame({"min_score": agg1, "max_score": agg2, "total": agg3, "Bads": agg4, "Mean_Score": agg5})
    ret_tbl["Goods"] = ret_tbl["total"] - ret_tbl["Bads"]
    ret_tbl["Bad_Rate"] = np.round(ret_tbl["Bads"] / ret_tbl["total"], 4) * 100
    ret_tbl = ret_tbl.sort_index(axis=0, ascending=False)
    ret_tbl.index = range(n_bkts)

    ret_tbl["Cum_Bad"] = np.round((ret_tbl.Bads / ret_tbl.Bads.sum()).cumsum(), 4) * 100
    ret_tbl["Cum_Good"] = np.round((ret_tbl.Goods / ret_tbl.Goods.sum()).cumsum(), 4) * 100
    ret_tbl['KS'] = np.abs(ret_tbl["Cum_Bad"] - ret_tbl["Cum_Good"])

    gini = ret_tbl["Cum_Bad"][0] * ret_tbl["Cum_Good"][0] / (2 * 100 * 100)
    for i in range(1, n_bkts):
        gini += (ret_tbl["Cum_Bad"][i] + ret_tbl["Cum_Bad"][i - 1]) * (ret_tbl["Cum_Good"][i] -
                                                                       ret_tbl["Cum_Good"][i - 1]) / (2 * 100 * 100)
    gini = np.round(2. * gini - 1., 4) * 100

    ks = np.round(ret_tbl.KS.max(), 2)

    coltitles = ["min_score", "max_score", "Mean_Score", "total", "Goods", "Bads", "Bad_Rate", "Cum_Bad", "Cum_Good", "KS"]

    return ret_tbl[coltitles], {"Gini": gini, "KS": ks}

def rnkOrdering2_con_wt(data_org, bad_col, wgt_col, score_col, n_bkts):
    if wgt_col is None:
        wgt_col = 'wt'
        #data_org[wgt_col] = 1
    #data_org[score_col] = -1 * data_org[score_col]
    data = data_org.sort_values(by=[score_col]).reset_index()
    data["cum_wgt"] = data[wgt_col].cumsum()
    total = data[wgt_col].sum()
    bin_lvl = 0.2
    flag = 1
    data.reset_index(inplace=True)
    for i in range(len(data)):
        if (data.loc[i, "cum_wgt"] >= total * bin_lvl) and (bin_lvl < 0.9):
            bin_lvl = bin_lvl + 0.2
            flag = flag + 1
            data.loc[i, "bucket"] = flag
        else:
            data.loc[i, "bucket"] = flag
#     data['bucket'] = pd.qcut(data[score_col], n_bkts)
    grouped = data.groupby('bucket', as_index=True)
    data["labels"] = data[bad_col] * data[wgt_col]
    agg1 = grouped[score_col].min()
    agg2 = grouped[score_col].max()
    agg3 = grouped[wgt_col].sum()
    agg4 = grouped["labels"].sum()
    agg5 = np.round(grouped[score_col].mean() * 100, 2)
    ret_tbl = pd.DataFrame({"min_score": agg1, "max_score": agg2, "total": agg3, "Bads": agg4, "Mean_Score": agg5})
    ret_tbl["Goods"] = ret_tbl["total"] - ret_tbl["Bads"]
    ret_tbl["Bad_Rate"] = np.round(ret_tbl["Bads"] / ret_tbl["total"], 4) * 100
    ret_tbl = ret_tbl.sort_index(axis=0, ascending=False)
    ret_tbl.index = range(n_bkts)

    ret_tbl["Cum_Bad"] = np.round((ret_tbl.Bads / ret_tbl.Bads.sum()).cumsum(), 4) * 100
    ret_tbl["Cum_Good"] = np.round((ret_tbl.Goods / ret_tbl.Goods.sum()).cumsum(), 4) * 100
    ret_tbl['KS'] = np.abs(ret_tbl["Cum_Bad"] - ret_tbl["Cum_Good"])

    gini = ret_tbl["Cum_Bad"][0] * ret_tbl["Cum_Good"][0] / (2 * 100 * 100)
    for i in range(1, n_bkts):
        gini += (ret_tbl["Cum_Bad"][i] + ret_tbl["Cum_Bad"][i - 1]) * (ret_tbl["Cum_Good"][i] -
                                                                       ret_tbl["Cum_Good"][i - 1]) / (2 * 100 * 100)
    gini = np.round(2. * gini - 1., 4) * 100

    ks = np.round(ret_tbl.KS.max(), 2)

    coltitles = ["min_score", "max_score", "Mean_Score", "total", "Goods", "Bads", "Bad_Rate", "Cum_Bad", "Cum_Good", "KS"]

    return ret_tbl[coltitles], {"Gini": gini, "KS": ks}

def rnkOrdering2_ks_code(data_org, bad_col, wgt_col, score_col, n_bkts):
    if wgt_col is None:
        wgt_col = 'wt'
        #data_org[wgt_col] = 1
    #data_org[score_col] = -1 * data_org[score_col]
    data = data_org.sort_values(by=[score_col]).reset_index()
    data["cum_wgt"] = data[wgt_col].cumsum()
    total = data[wgt_col].sum()
    bin_lvl = 0.2
    flag = 1
    data.reset_index(inplace=True)
    for i in range(len(data)):
        if (data.loc[i, "cum_wgt"] >= total * bin_lvl) and (bin_lvl < 0.9):
            bin_lvl = bin_lvl + 0.2
            flag = flag + 1
            data.loc[i, "bucket"] = flag
        else:
            data.loc[i, "bucket"] = flag
#     data['bucket'] = pd.qcut(data[score_col], n_bkts)
    grouped = data.groupby('bucket', as_index=True)
    data["labels"] = data[bad_col] * data[wgt_col]
    agg1 = grouped[score_col].min()
    agg2 = grouped[score_col].max()
    agg3 = grouped[wgt_col].sum()
    agg4 = grouped["labels"].sum()
    agg5 = np.round(grouped[score_col].mean() * 100, 2)
    ret_tbl = pd.DataFrame({"min_score": agg1, "max_score": agg2, "total": agg3, "Bads": agg4, "Mean_Score": agg5})
    ret_tbl["Goods"] = ret_tbl["total"] - ret_tbl["Bads"]
    ret_tbl["Bad_Rate"] = np.round(ret_tbl["Bads"] / ret_tbl["total"], 4) * 100
    ret_tbl = ret_tbl.sort_index(axis=0, ascending=False)
    ret_tbl.index = range(n_bkts)

    ret_tbl["Cum_Bad"] = np.round((ret_tbl.Bads / ret_tbl.Bads.sum()).cumsum(), 4) * 100
    ret_tbl["Cum_Good"] = np.round((ret_tbl.Goods / ret_tbl.Goods.sum()).cumsum(), 4) * 100
    ret_tbl['KS'] = np.abs(ret_tbl["Cum_Bad"] - ret_tbl["Cum_Good"])

    gini = ret_tbl["Cum_Bad"][0] * ret_tbl["Cum_Good"][0] / (2 * 100 * 100)
    for i in range(1, n_bkts):
        gini += (ret_tbl["Cum_Bad"][i] + ret_tbl["Cum_Bad"][i - 1]) * (ret_tbl["Cum_Good"][i] -
                                                                       ret_tbl["Cum_Good"][i - 1]) / (2 * 100 * 100)
    gini = np.round(2. * gini - 1., 4) * 100

    ks = np.round(ret_tbl.KS.max(), 2)

    coltitles = ["min_score", "max_score", "Mean_Score", "total", "Goods", "Bads", "Bad_Rate", "Cum_Bad", "Cum_Good", "KS"]

    return  ks,gini


# Given by Kushal Gandhi

from pandas.util import hash_pandas_object

def CSI_decile(dataset,var,monitoring_df):
    dataset['Quantile_rank']=pd.qcut(dataset[var],10,labels=False,duplicates='drop')
    summary= dataset.groupby('Quantile_rank').agg({var:['min','max','mean','count']})
    summary.columns=summary.columns.droplevel()
    summary=summary.rename(columns={'min':'Min_'+var+'_dev',
                               'max':'Max_'+var+'_dev',
                               'mean':'Mean_'+var+'_dev',
                               'count':'Total_dev'}).reset_index()
    try:
        monitoring_df.loc[((monitoring_df[var]<summary.iloc[1,1] )),'Quantile_rank']=0
        monitoring_df.loc[((monitoring_df[var]>= summary.iloc[1,1]) & (monitoring_df[var]<summary.iloc[2,1])),'Quantile_rank']=1
        monitoring_df.loc[((monitoring_df[var]>= summary.iloc[2,1]) & (monitoring_df[var]<summary.iloc[3,1])),'Quantile_rank']=2
        monitoring_df.loc[((monitoring_df[var]>= summary.iloc[3,1]) & (monitoring_df[var]<summary.iloc[4,1])),'Quantile_rank']=3
        monitoring_df.loc[((monitoring_df[var]>= summary.iloc[4,1]) & (monitoring_df[var]<summary.iloc[5,1])),'Quantile_rank']=4
        monitoring_df.loc[((monitoring_df[var]>= summary.iloc[5,1]) & (monitoring_df[var]<summary.iloc[6,1])),'Quantile_rank']=5
        monitoring_df.loc[((monitoring_df[var]>= summary.iloc[6,1]) & (monitoring_df[var]<summary.iloc[7,1])),'Quantile_rank']=6
        monitoring_df.loc[((monitoring_df[var]>= summary.iloc[7,1]) & (monitoring_df[var]<summary.iloc[8,1])),'Quantile_rank']=7
        monitoring_df.loc[((monitoring_df[var]>= summary.iloc[8,1]) & (monitoring_df[var]<summary.iloc[9,1])),'Quantile_rank']=8
        monitoring_df.loc[(monitoring_df[var]>= summary.iloc[9,1]),'Quantile_rank']=9
    except:
        monitoring_df.Quantile_rank.fillna(len(summary)-1, inplace=True)
    outsample=monitoring_df.groupby('Quantile_rank').agg({var: ['min', 'max','mean','count']})
    outsample=pd.DataFrame(outsample)
    outsample.columns = ['min', 'max','mean','count']
    outsample=outsample.rename(columns={'min':'Min_'+var+'_monitor','max':'Max_'+var+'_monitor','mean':'Mean_'+var+'_monitor','count':'Total_monitor'})
    outsample=outsample.reset_index()
    final_df=pd.merge(summary,outsample,on='Quantile_rank',how='left')
    final_df.head()
    final_df['prop_train']=final_df['Total_dev']/len(dataset)
    final_df['prop_outsamp']=final_df['Total_monitor']/len(monitoring_df)
    final_df['RAW_CSI']=(final_df['prop_train']-final_df['prop_outsamp'])*np.log(final_df['prop_train']/final_df['prop_outsamp'])
    final_df['CUMSUM_CSI'] = final_df['RAW_CSI'].cumsum()
    CSI=np.round(final_df.CUMSUM_CSI.max(), 5)
#     display(final_df)
    return CSI#,final_df

"""
def CSI_pentile(dataset,var,monitoring_df):
    dataset['Quantile_rank']=pd.qcut(dataset[var],5,labels=False,duplicates='drop')
    summary= dataset.groupby('Quantile_rank').agg({var:['min','max','mean','count']})
    summary.columns=summary.columns.droplevel()
    summary=summary.rename(columns={'min':'Min_'+var+'_dev',
                               'max':'Max_'+var+'_dev',
                               'mean':'Mean_'+var+'_dev',
                               'count':'Total_dev'}).reset_index()
#         print(summary)
#         print(summary.shape[0])
    summ_shape=summary.shape[0]
    if summ_shape==5:
        monitoring_df.loc[((monitoring_df[var]<summary.iloc[1,1] )),'Quantile_rank']=0
        monitoring_df.loc[((monitoring_df[var]>= summary.iloc[1,1])&(monitoring_df[var]<summary.iloc[2,1])),'Quantile_rank']=1
        monitoring_df.loc[((monitoring_df[var]>= summary.iloc[2,1]) &(monitoring_df[var]<summary.iloc[3,1])),'Quantile_rank']=2
        monitoring_df.loc[((monitoring_df[var]>= summary.iloc[3,1]) &(monitoring_df[var]<summary.iloc[4,1])),'Quantile_rank']=3
        monitoring_df.loc[((monitoring_df[var]>= summary.iloc[4,1])),'Quantile_rank']=4
    elif summ_shape==4:
        monitoring_df.loc[((monitoring_df[var]<summary.iloc[1,1] )),'Quantile_rank']=0
        monitoring_df.loc[((monitoring_df[var]>= summary.iloc[1,1])&(monitoring_df[var]<summary.iloc[2,1])),'Quantile_rank']=1
        monitoring_df.loc[((monitoring_df[var]>= summary.iloc[2,1]) &(monitoring_df[var]<summary.iloc[3,1])),'Quantile_rank']=2
        monitoring_df.loc[((monitoring_df[var]>= summary.iloc[3,1])),'Quantile_rank']=3
    elif summ_shape==3:
        monitoring_df.loc[((monitoring_df[var]<summary.iloc[1,1] )),'Quantile_rank']=0
        monitoring_df.loc[((monitoring_df[var]>= summary.iloc[1,1])&(monitoring_df[var]<summary.iloc[2,1])),'Quantile_rank']=1
        monitoring_df.loc[((monitoring_df[var]>= summary.iloc[2,1])),'Quantile_rank']=2
    elif summ_shape==2:
        monitoring_df.loc[((monitoring_df[var]<summary.iloc[1,1] )),'Quantile_rank']=0
        monitoring_df.loc[((monitoring_df[var]>= summary.iloc[1,1])),'Quantile_rank']=1
    else: 
        csi_cust,final_df1=CSI_custom(dataset,var,monitoring_df,var+'_bins')
        summ_shape=1
        return csi_cust,final_df1,summ_shape
        pass
    outsample=monitoring_df.groupby('Quantile_rank').agg({var: ['min', 'max','mean','count']})
    outsample=pd.DataFrame(outsample)
    outsample.columns = ['min', 'max','mean','count']
    outsample=outsample.rename(columns={'min':'Min_'+var+'_monitor','max':'Max_'+var+'_monitor','mean':'Mean_'+var+'_monitor','count':'Total_monitor'})
    outsample=outsample.reset_index()
    final_df=pd.merge(summary,outsample,on='Quantile_rank',how='left')
    final_df.head()
    final_df['prop_train']=final_df['Total_dev']/len(dataset)
    final_df['prop_outsamp']=final_df['Total_monitor']/len(monitoring_df)
    final_df['RAW_CSI']=(final_df['prop_train']-final_df['prop_outsamp'])*np.log(final_df['prop_train']/final_df['prop_outsamp'])
    final_df['CUMSUM_CSI'] = final_df['RAW_CSI'].cumsum()
    CSI=np.round(final_df.CUMSUM_CSI.max(), 5)
    return CSI,final_df,summ_shape
"""
# CSI_decile(dev_df,'subjectage',monitoring_df)
