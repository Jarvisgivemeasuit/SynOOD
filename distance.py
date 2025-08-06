
from adaptesting import tst # Load the main library to conduct tst

# Load mnist data as example, make sure the input data should be Pytorch Tensor
import torch
import torchvision
import torchvision.transforms as transforms
import random
import time
import pickle
import os
from PIL import Image

start = time.time()
torch.manual_seed(0)
random.seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Distribution P
# data_fake_all = pickle.load(
#     open('Fake_MNIST_data_EP100_N10000.pckl', 'rb'))[0]
# data_fake_all = torch.from_numpy(data_fake_all)  # Convert to tensor


fake_img_folder = 'ood_generated_samples/near_energy'
classes = os.listdir(fake_img_folder)
fake_img_paths = []
related_paths = []
for cls in classes:
    fake_img_paths += [os.path.join(fake_img_folder, cls, fname) for fname in os.listdir(os.path.join(fake_img_folder, cls))  if fname.endswith('.JPEG')]
    related_paths += [os.path.join(cls, fname) for fname in os.listdir(os.path.join(fake_img_folder, cls))  if fname.endswith('.JPEG')]


transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

data_fake_all = torch.stack([
    transform(Image.open(img_path).convert('L'))
    for img_path in fake_img_paths
])

# Distribution Q
transform = transforms.Compose(
    [transforms.Resize(32),  # Make sure the dimension of P and Q are same
     transforms.ToTensor(),  # Convert to tensor
     transforms.Normalize([0.5], [0.5])]
)
real_img_folder = '/mnt/share/cs22-hongly/DATACENTER/ImageNet/train'
real_img_paths = [os.path.join(real_img_folder, file.replace("_ood", '')) for file in related_paths]
data_real_all = torch.stack([
    transform(Image.open(img_path).convert('L'))
    for img_path in real_img_paths
])

Z1 = data_fake_all
Z2 = data_real_all # Test power
# Z2 = data_fake_all  # Type-I error

counter = 0
n_trial = 100
n_samples = 250

# Conduct Experiments for n_trial times,
# remove the for loop if only want to get a result of reject or not
for _ in range(n_trial):

    # Create a list of indices from 0 to 199
    indices = list(range(10000))

    # Shuffle the indices
    random.shuffle(indices)

    # Select the first 100 shuffled indices for X
    X_indices = indices[:n_samples]

    # Select the remaining indices for Y
    Y_indices = indices[n_samples:n_samples * 2]

    # Sample X and Y from Z using the selected indices
    X = Z1[X_indices]
    # Y = Z2[X_indices]
    Y = Z2[Y_indices]

    # Five kinds of SOTA TST methods to choose
    h, _, _ = tst(X, Y, device=device)  # default method is median heuristic
    # h, _, _ = tst(X, Y, device=device, method="fuse", kernel="laplace_gaussian", n_perm=2000)
    # h, _, _ = tst(X, Y, device=device, method="agg", n_perm=3000)
    # h, _, _ = tst(X, Y, device=device, method="clf", data_type="image", patience=150, n_perm=200)
    # h, _, _ = tst(X, Y, device=device, method="deep", data_type="image", patience=150, n_perm=200)
    counter += h

print(f"Power: {counter}/{n_trial}")
end = time.time()
print(f"Time taken: {end - start:.4f} seconds")