import model

import preprocess
import os
import pandas as pd
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
    path = '/Data2/ghjoo/data/lbrbafmi/lbrbafmi/data_lbrbaf/df_lbrbaf_all.csv'
    df = pd.read_csv(path)
    df_test = df[(df['tvt']=='test')]
    x = df_test['filename'].values
    y = df_test[df_test.columns[1:-1]].values
    

    save_path='./save_h5'
    data, data_label = preprocess.load_data(x,y)
    print(data.shape)
    print(data_label.shape)
    model.ekgan_train(data,data_label,50,save_path,1)
