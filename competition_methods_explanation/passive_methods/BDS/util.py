

def table_to_binary_df(table,target_class='hit'):
    import Orange
    from Orange.data.pandas_compat import table_to_frame
    import pandas as pd
    disc = Orange.preprocess.Discretize()
    disc.method = Orange.preprocess.discretize.EqualFreq(n=3)
    # disc.method = Orange.preprocess.discretize.EntropyMDL()
    disc_data_table = disc(table)
    df = table_to_frame(disc_data_table)
    # Y = pd.DataFrame(disc_data_table.Y,columns=[disc_data_table.domain.class_var.name],dtype='int32')
    Y = disc_data_table.Y
    df.drop(df.columns[-1],axis = 1, inplace = True)
    df = pd.get_dummies(df)
    return disc_data_table,df,Y
