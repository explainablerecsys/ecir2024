from multiprocessing import Pool
from datasets import Dataset
from os import listdir
from os.path import isfile, join
import pandas as pd
from pathlm.utils import get_eid_to_name_map, get_rid_to_name_map


class PathDataset:
    def __init__(self, dataset_name: str, base_data_dir: str="", task: str=None, sample_size: str=None, n_hop: str=None, plain_text_path=False):
        self.dataset_name = dataset_name
        self.base_data_dir = base_data_dir
        self.data_dir = join(self.base_data_dir, "paths_random_walk")
        self.task = task
        self.sample_size = sample_size
        self.n_hop = n_hop
        self.plain_text_path = plain_text_path #Currently not used, experimental parameter
        self.read_single_csv_to_hf_dataset()
        # Get eid2name and rid2name
        self.eid2name = get_eid_to_name_map(self.dataset_name)
        self.rid2name = get_rid_to_name_map(self.dataset_name)



    # Based on the path struct, for now it is p to p
    def convert_numeric_path_to_textual_path(self, path: str) -> str:
        path_list = path.split(" ")
        ans = []
        for pos, token in enumerate(path_list):
            # Handle user and watched relation
            if pos == 0:
                ans.append(f"U{token}")
            elif pos == 1:
                ans.append(token)
            # Handle recommendation
            elif pos == 2 or pos == 6 or pos == 10:
                #ans.append("<recommendation>")
                ans.append(self.eid2name[token])
            # Handle entity
            elif pos % 2 == 0:
                ans.append(self.eid2name[token])
            # Handle relation
            else:
                ans.append(self.rid2name[token])
            ans.append("<word_end>")
        return " ".join(ans)

    def keep_numeric(self, path: str) -> str:
        path_list = path.split(" ")
        ans = []
        for pos, token in enumerate(path_list):
            # Handle user
            if pos == 0:
                ans.append(f"U{token}")
            elif pos == 1:
                ans.append(token)
            # Handle recommendation
            elif pos == 2 or pos == 6 or pos == 10:
                ans.append(f"P{token}")
            # Handle entity
            elif pos % 2 == 0:
                ans.append(f"E{token}")
            # Handle relation
            else:
                ans.append(f"R{token}")
        return " ".join(ans)

    def read_csv_as_dataframe(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(join(self.data_dir, filename), header=None, names=["path"], index_col=None)

    def read_multiple_csv_to_hf_dataset(self):
        file_list = [f for f in listdir(self.data_dir) if isfile(join(self.data_dir, f))]

        # set up your pool
        #with Pool(processes=8) as pool:  # orc whatever your hardware can support
        #    df_list = pool.map(self.read_csv_as_dataframe, file_list)#
        #
        #    # reduce the list of dataframes to a single dataframe
        df_list = []
        for filename in file_list:
            df_list.append(self.read_csv_as_dataframe(filename))

        combined_df = pd.concat(df_list, ignore_index=True)


        # Convert to HuggingFace Dataset
        self.dataset = Dataset.from_pandas(combined_df)

    def read_single_csv_to_hf_dataset(self) -> None:
        #file_list = [f for f in listdir(self.data_dir) if isfile(join(self.data_dir, f))]
        #filename = f'paths_{self.task}_{self.sample_size}_{self.n_hop}.txt'
        filename = f'paths_{self.task}_{self.sample_size}_{self.n_hop}.txt'
        #filepath = join(self.data_dir, filename)


        df = self.read_csv_as_dataframe(filename)
        self.dataset = Dataset.from_pandas(df)
        
        #for filename in file_list:
        #    print(filename)
        #    if filename == f'paths_{self.task}_{self.sample_size}_{self.n_hop}.txt':#
        #
        #        df = self.read_csv_as_dataframe(filename)
        #        self.dataset = Dataset.from_pandas(df)
        #        continue



    def show_random_examples(self) -> None:
        print(self.dataset["path"][:10])