import os
import subprocess
import torch
import shutil
import sys
sys.path.append("d:/git/dfodl/")
sys.path.append("d:/git/dfodl/yolov5")

def train_local_yolov5(
    node_id: int,
    round_id: int,
    data_yaml: str,
    base_model_path: str,
    output_dir: str,
    epochs: int = 5,
    yolov5_dir: str = "yolov5",
    device_str: str = "0"
):
    """
    Trains YOLOv5 locally (single process) for one node/round.
    - If base_model_path == "", YOLOv5 trains from scratch (random init).
    - 'device_str' can be "0", "cuda:0", "cpu", or "0,1" (multi-GPU), etc.
    
    :param node_id: Node index
    :param round_id: Federated round index
    :param data_yaml: Path to this node's data config (e.g. 'subset_1/data_node0.yaml')
    :param base_model_path: Model weights to load ("" => from scratch)
    :param output_dir: Where to store final model
    :param epochs: Number of epochs to train
    :param yolov5_dir: Path to YOLOv5 code (containing 'train.py')
    :param device_str: Which device to use for training
    :return: Path to the final model file from this node's training
    """
    node_round_run_dir = os.path.join(output_dir, f"node{node_id}_round{round_id}")
    os.makedirs(node_round_run_dir, exist_ok=True)

    train_py = os.path.join(yolov5_dir, "train.py")
    cfg_yaml = yolov5_dir + "/models/yolov5n.yaml"
    print(cfg_yaml)
    # YOLOv5 command
    if base_model_path == None or base_model_path=='':
        command = [
        "D:/git/dfodl/.venv/Scripts/python.exe", train_py,
        "--data", data_yaml,
        "--weights", base_model_path,  # "" => random init from scratch
        "--epochs", str(epochs),
        "--batch-size", "64",
        "--device", device_str,        # e.g. "0", "cuda:0", or "cpu"
        "--project", node_round_run_dir,
        "--name", "exp",
        "--cfg", cfg_yaml,
        "--exist-ok",
        ]
    else:
        
        command = [
            "D:/git/dfodl/.venv/Scripts/python.exe", train_py,
            "--data", data_yaml,
            "--weights", base_model_path,  # "" => random init from scratch
            "--epochs", str(epochs),
            "--batch-size", "64",
            "--device", device_str,        # e.g. "0", "cuda:0", or "cpu"
            "--project", node_round_run_dir,
            "--name", "exp",
            # "--cfg", cfg_yaml,
            "--exist-ok",
        ]
    
    print(f"[Node {node_id}, Round {round_id}] Training on device {device_str}, base_model_path={repr(base_model_path)}")
    print("command: ", command)
    subprocess.run(command, check=True)

    # YOLOv5 typically saves last.pt in node_round_run_dir/exp/weights/
    last_pt = os.path.join(node_round_run_dir, "exp", "weights", "last.pt")
    if not os.path.exists(last_pt):
        raise FileNotFoundError(f"[Node {node_id}, Round {round_id}] Did not find {last_pt} after training")

    # Rename it to a more descriptive name
    final_model_path = os.path.join(node_round_run_dir, f"model_round{round_id}_node{node_id}.pt")
    shutil.move(last_pt, final_model_path)
    
    print(f"[Node {node_id}, Round {round_id}] Finished training => {final_model_path}")
    return final_model_path


def load_weights_and_avg(model_paths, owner_idx=0):
    """
    Loads multiple YOLOv5 .pt files, performing partial FedAvg on only
    the parameters that end with '.weight' or '.bias'.
    
    All other parameters (e.g. BatchNorm running_mean, num_batches_tracked, 
    or any custom parameters) are kept from the 'owner' checkpoint (owner_idx).
    
    :param model_paths: List of checkpoint paths for the group (neighbors + self).
    :param owner_idx: Index in 'model_paths' corresponding to the node's own model.
                     We'll use that checkpoint as the "owner" for non-(weight|bias) params.
    :return: aggregated_state_dict
    """   
    # 1) Load all models
    all_sd = []
    for mp in model_paths:
        ckpt = torch.load(mp, map_location="cpu", weights_only=False)
        sd = ckpt["model"].state_dict()
        all_sd.append(sd)
    
    # 2) Use the owner node's checkpoint as the base
    base_sd = {k: v.clone() for k, v in all_sd[owner_idx].items()}
    
    # We'll sum up the weights/bias from all models, then average them.
    # For non-weight/bias params, we keep the owner's version.
    
    count = len(model_paths)
    
    # 3) Iterate over every parameter in base_sd
    for param_name in base_sd.keys():
        if param_name.endswith(".weight") or param_name.endswith(".bias"):
            # This is a parameter we want to average.
            summed = None
            for sd in all_sd:
                if summed is None:
                    summed = sd[param_name].clone()
                else:
                    summed += sd[param_name]
            base_sd[param_name] = summed / count
        
        else:
            # This param is not .weight or .bias => keep the owner's param
            # (base_sd already set to the owner version, so we do nothing)
            pass
    
    return base_sd

def save_averaged_weights(avg_state_dict, ref_model_path, out_path):
    """
    Saves the averaged weights into a new YOLOv5 .pt file, 
    using ref_model_path for structure (hyperparameters, etc.).
    
    :param avg_state_dict: FedAvg result
    :param ref_model_path: YOLOv5 .pt to copy extra fields from
    :param out_path: Where to save the aggregated model
    """
    ckpt = torch.load(ref_model_path, map_location="cpu", weights_only=False)
    ckpt["model"].load_state_dict(avg_state_dict)
    torch.save(ckpt, out_path)
    print(f"[Aggregation] Saved averaged model => {out_path}")


def run_decentralized_fedavg(
    subsets_dir="coco_splits",
    num_nodes=10,
    rounds=5,
    local_epochs=5,
    from_scratch_first_round=True,
    init_model=None,            # If from_scratch_first_round=False, supply a model for round 0
    output_dir="dfl_output",
    topology=None,              # e.g. ring or star graph
    yolov5_dir="yolov5",
    device_str="0"
):
    """
    Decentralized FedAvg for YOLOv5:
      - If from_scratch_first_round=True, then the 1st round (round 0) uses random init (weights="").
      - Otherwise, if from_scratch_first_round=False, the 1st round uses init_model.
      - Each node trains local_epochs, then aggregates with neighbors, 
        producing new weights for the next round.

    :param subsets_dir: Directory containing subset_1..subset_n, each with data_node{i}.yaml
    :param num_nodes: Number of nodes
    :param rounds: Number of federated rounds
    :param local_epochs: Training epochs each round at each node
    :param from_scratch_first_round: If True => round 0 trains from scratch, else uses init_model
    :param init_model: A pretrained .pt if not training from scratch (only used if from_scratch_first_round=False)
    :param output_dir: Where to store the entire run's output
    :param topology: adjacency list, topology[i] = neighbors of node i
    :param yolov5_dir: Path to YOLOv5 code directory
    :param device_str: GPU/CPU spec, e.g. "0" for T4 GPU, "cpu" for CPU, "cuda:1" for second GPU, etc.
    """
    os.makedirs(output_dir, exist_ok=True)

    # If no topology provided, default to ring
    if topology is None:
        topology = []
        for i in range(num_nodes):
            left = (i - 1) % num_nodes
            right = (i + 1) % num_nodes
            topology.append([left, right])

    # current_model_paths[i] is the model that node i will load *this round*
    # We'll do round indexing: r in [0..rounds-1].
    # Round 0: either from scratch or from init_model.
    if from_scratch_first_round:
        current_model_paths = [""] * num_nodes
    else:
        if init_model is None:
            raise ValueError("Please provide 'init_model' if from_scratch_first_round=False.")
        current_model_paths = [init_model] * num_nodes

    for r in range(rounds):
        print(f"\n===== Federated Round {r} =====")

        # (a) local training
        round_models = []
        for node_id in range(num_nodes):
            # each node has data_node{node_id}.yaml in subset_{node_id+1} 
            subset_id = node_id + 1
            data_yaml = os.path.join(subsets_dir, f"subset_{subset_id}", f"data_node{node_id}.yaml")

            # train
            trained_model_path = train_local_yolov5(
                node_id=node_id,
                round_id=r,
                data_yaml=data_yaml,
                base_model_path=current_model_paths[node_id],
                output_dir=output_dir,
                epochs=local_epochs,
                yolov5_dir=yolov5_dir,
                device_str=device_str
            )
            round_models.append(trained_model_path)

        # (b) decentralized aggregation
        new_paths = [None] * num_nodes
        for node_id in range(num_nodes):
            # neighbors => local adjacency
            neighbors = topology[node_id]
            group_ids = [node_id] + neighbors
            group_model_paths = [round_models[g] for g in group_ids]
            owner_idx=0
            avg_sd = load_weights_and_avg(group_model_paths, owner_idx)
            ref_path = round_models[node_id]
            agg_path = os.path.join(output_dir, f"node{node_id}_round{r}", f"agg_model_round{r}_node{node_id}.pt")
            save_averaged_weights(avg_sd, ref_path, agg_path)

            new_paths[node_id] = agg_path

        # (c) update for next round
        current_model_paths = new_paths

    print(f"\n[Done] Decentralized FedAvg finished. Check '{output_dir}' for outputs.")


# Example usage
if __name__ == "__main__":
    # Suppose you have 10 subsets in 'coco_splits/subset_1..subset_10',
    # each containing data_node0.yaml..data_node9.yaml, etc.
    # and you want to train on an NVIDIA T4 GPU (device 0).
    
    run_decentralized_fedavg(
        subsets_dir="D:/git/dfodl/datasets/coco_split/",
        num_nodes=10,
        rounds=10,
        local_epochs=10,
        from_scratch_first_round=True,      # Round 0 => random init
        init_model=None,                    # Not used since from_scratch_first_round=True
        output_dir="dfl_output",
        topology=None,                      # ring by default
        yolov5_dir="D:/git/dfodl/yolov5",
        device_str="0"                      # T4 typically at device "0"
    )
