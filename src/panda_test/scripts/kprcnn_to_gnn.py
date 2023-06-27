import torch
from torch_geometric.data import Data, Dataset

class RoboticArmDataset(Dataset):
    def __init__(self, raw_data_list, transform=None, pre_transform=None):
        self.raw_data_list = raw_data_list
        super(RoboticArmDataset, self).__init__(".", transform, pre_transform)

    def len(self):
        return len(self.raw_data_list)

    def get(self, idx):
        item = self.raw_data_list[idx]
        
        # Use the keypoint coordinates as node features
        x = torch.tensor(item['keypoints'], dtype=torch.float)
        
        # Construct the edges based on your domain knowledge
        # Here, we'll assume that keypoints are connected in the order they appear in the keypoints list
        edge_index = torch.tensor([[i, i+1] for i in range(len(item['keypoints'])-1)], dtype=torch.long)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # to make edges undirected

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index.t().contiguous())

        return data