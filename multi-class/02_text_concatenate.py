import pandas as pd
import os
import time

my_folder="s3://trident-retention-output/"

#### Askunum text data ####
askunum_text=pd.DataFrame()
for year in [2018,2019,2020,2021,2022]:
    new_data=pd.read_csv(os.path.join(my_folder,f"askunum_textbody_{year}"+".csv"))
    askunum_text=pd.concat([askunum_text,new_data])
    print("{:<15}{:<20,}".format(year,new_data.shape[0]))
    
askunum_text.drop(['Unnamed: 0'],axis=1,inplace=True)
askunum_text['unum_id']=askunum_text['unum_id'].astype(int).astype(str)
askunum_text.sort_values(["unum_id","year","month","MessageDate"],inplace=True,ascending=True)
askunum_text.to_pickle(os.path.join(my_folder,"askunum_text_pickle"))

start=time.time()
askunum_text=askunum_text.groupby(["ParentId","Subtype"])['TextBody'].apply(lambda x: " ".join(x)).reset_index()
askunum_text=askunum_text.drop_duplicates()
end=time.time()
print("It take {:.4f} second to group data".format(end-start))

askunum_text.to_pickle(os.path.join(my_folder,"askunum_text_v1"))
