import sys
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from fs_mol.data.mat import get_mat_batcher, mat_process_samples
from fs_mol.utils.torch_utils import torchify
from fs_mol.utils.metrics import r2_score_os

from fs_mol.models.mat import MATModel
from fs_mol.models.abstract_torch_fsmol_model import load_model_weights

from botorch.optim.fit import fit_gpytorch_scipy

from bayes_opt.bo_utils import load_antibiotics_dataset, load_covid_moonshot_dataset, load_dockstring_dataset, load_cep_dataset, run_gp_ei_bo, min_so_far, task_to_batches, create_gp
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_name = sys.argv[1]
if task_name == "anti":
    task = load_antibiotics_dataset("../antibiotics/antibiotics-dataset.xlsx", "../../fs_mol/preprocessing/utils/helper_files/")
elif task_name == "covid":
    task = load_covid_moonshot_dataset("../covid_moonshot/covid-moonshot-dataset.csv", "../../fs_mol/preprocessing/utils/helper_files/")
elif task_name == "dock":
    task = load_dockstring_dataset("../dockstring/dockstring-dataset-subsampled.csv", "../../fs_mol/preprocessing/utils/helper_files/")
elif task_name == "opv":
    task = load_cep_dataset("../organic_photovoltaics/cep-dataset-subsampled.csv", "../../fs_mol/preprocessing/utils/helper_files/")
else:
    raise ValueError

mat_samples = mat_process_samples(task.samples)

batcher = get_mat_batcher(max_num_graphs=3)
mat_batches = torchify(
    task_to_batches(None, batcher, mat_samples), 
    device=torch.device("cpu")
)

model_weights_file = "../../../fs-mol-checkpoints/mat_pretrained_weights.pt"

mat_model = MATModel.build_from_model_file(model_weights_file, quiet=True, device=device, config_overrides={"num_tasks": 1})
load_model_weights(mat_model, model_weights_file, load_task_specific_weights=False)

mat_model.eval()

representations = []

with torch.no_grad():
    for features in mat_batches:
        node_features = features.node_features.to(device)
        mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        representation = mat_model.encode(node_features, mask, features.adjacency_matrix.to(device), features.distance_matrix.to(device), None)

        mask = mask.unsqueeze(-1).float()
        out_masked = representation * mask
        out_sum = out_masked.sum(dim=1)
        mask_sum = mask.sum(dim=(1))
        out_avg_pooling = out_sum / mask_sum

        representations.append(out_avg_pooling.cpu())
    
del mat_model
torch.cuda.empty_cache()

dataset = task.samples

x_all = torch.cat(representations, dim=0)
y_all = torch.FloatTensor([float(x.numeric_label) for x in dataset]).to(device)

stratified_space = x_all.shape[0] // int(sys.argv[2])

r2s = []
nll = []
for i in range(200):
    #training_data_idx = [random.randrange(i*stratified_space, (i+1)*stratified_space) for i in range(0, int(sys.argv[2]))]
    if i < 100:
        training_data_idx = np.random.choice(np.arange(x_all.shape[0]), size=int(sys.argv[2]), replace=False).tolist()
    else:
        training_data_idx = np.random.choice(np.arange(x_all.shape[0]), size=int(sys.argv[3]), replace=False).tolist()
    test_data_idx = [i for i in range(len(dataset)) if i not in training_data_idx]
    x_train = x_all[training_data_idx]
    y_train = y_all[training_data_idx]
    x_test = x_all[test_data_idx]
    y_test = y_all[test_data_idx]

    y_train_mean = y_train.mean()
    y_train_std = y_train.std()
    y_train = (y_train - y_train_mean) / y_train_std
    y_test = (y_test - y_train_mean) / y_train_std

    likelihood, model, mll = create_gp(x_train, y_train, "matern", device, 0.01, True)
    model.train()
    likelihood.train()
    fit_gpytorch_scipy(mll)
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        y_test_pred = likelihood(model(x_test))

    r2s.append(r2_score_os(y_test.cpu().detach().numpy(), y_test_pred.mean.cpu().detach().numpy()))
    nll.append(-y_test_pred.log_prob(y_test).cpu().detach().numpy() / y_test.shape[0])

r2s = np.array(r2s)
nll = np.array(nll)
print()
print("MAT")
print(sys.argv[1], sys.argv[2])
print("R2:", r2s[:100].mean(), r2s[:100].std(), "(", r2s[:100].std()/np.sqrt(100) ,")")
print("NLL:", nll[:100].mean(), nll[:100].std(), "(", nll[:100].std()/np.sqrt(100) ,")")
print(sys.argv[1], sys.argv[3])
print("R2:", r2s[100:].mean(), r2s[100:].std(), "(", r2s[100:].std()/np.sqrt(100) ,")")
print("NLL:", nll[100:].mean(), nll[100:].std(), "(", nll[100:].std()/np.sqrt(100) ,")")
print(sys.argv[1], sys.argv[2], sys.argv[3])
print("R2:", r2s.mean(), r2s.std(), "(", r2s.std()/np.sqrt(200) ,")")
print("NLL:", nll.mean(), nll.std(), "(", nll.std()/np.sqrt(200) ,")")
print()