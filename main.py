import json
import os
import typing
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import wandb
from dataclasses_json import dataclass_json
from flytekit import Resources, task, workflow
from flytekit.types.file import PythonPickledFile
from torch import distributed as dist
from torch import nn, optim
from torchvision import datasets, transforms



NUM_BATCHES_TO_LOG = 10
LOG_IMAGES_PER_BATCH = 32



def wandb_setup():
    wandb.login()
    wandb.init(
        project="mnist-single-node-single-gpu",
        entity=os.environ.get("WANDB_USERNAME", "my-user-name"),
    )


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



def mnist_dataloader(batch_size, train=True, **kwargs):
    return torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=train,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()

    # hooks into the model to collect gradients and the topology
    wandb.watch(model)

    # loop through the training batches
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # device conversion
        optimizer.zero_grad()  # clear gradient
        output = model(data)  # forward pass
        loss = F.nll_loss(output, target)  # compute loss
        loss.backward()  # propagate the loss backward
        optimizer.step()  # update the model parameters

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

            # log epoch and loss
            wandb.log({"loss": loss, "epoch": epoch})


def log_test_predictions(images, labels, outputs, predicted, my_table, log_counter):
    """
    Convenience function to log predictions for a batch of test images
    """

    # obtain confidence scores for all classes
    scores = F.softmax(outputs.data, dim=1)

    # assign ids based on the order of the images
    for i, (image, pred, label, score) in enumerate(
        zip(*[x.cpu().numpy() for x in (images, predicted, labels, scores)])
    ):
        # add required info to data table: id, image pixels, model's guess, true label, scores for all classes
        my_table.add_data(f"{i}_{log_counter}", wandb.Image(image), pred, label, *score)
        if i == LOG_IMAGES_PER_BATCH:
            break



@dataclass_json
@dataclass
class Hyperparameters(object):
    """
    Args:
        backend: pytorch backend to use, e.g. "gloo" or "nccl"
        sgd_momentum: SGD momentum (default: 0.5)
        seed: random seed (default: 1)
        log_interval: how many batches to wait before logging training status
        batch_size: input batch size for training (default: 64)
        test_batch_size: input batch size for testing (default: 1000)
        epochs: number of epochs to train (default: 10)
        learning_rate: learning rate (default: 0.01)
    """

    backend: str = dist.Backend.GLOO
    sgd_momentum: float = 0.5
    seed: int = 1
    log_interval: int = 10
    batch_size: int = 64

    test_batch_size: int = 1000
    epochs: int = 10
    learning_rate: float = 0.01


TrainingOutputs = typing.NamedTuple(
    "TrainingOutputs",
    epoch_accuracies=typing.List[float],
    model_state=PythonPickledFile,
)



if os.getenv("SANDBOX") != "":
    print(f"SANDBOX ENV: '{os.getenv('SANDBOX')}'")

    mem = "100Mi"
    gpu = "0"
    storage = "500Mi"
else:
    print(f"SANDBOX ENV: '{os.getenv('SANDBOX')}'")

    mem = "3Gi"
    gpu = "1"
    storage = "1Gi"


@task(
    retries=2,
    cache=True,
    cache_version="1.0",
    requests=Resources(gpu=gpu, mem=mem, storage=storage),
    limits=Resources(gpu=gpu, mem=mem, storage=storage),
)
def pytorch_mnist_task(hp: Hyperparameters) -> TrainingOutputs:
    wandb_setup()

    # store the hyperparameters' config in ``wandb``
    wandb.config.update(json.loads(hp.to_json()))

    # set random seed
    torch.manual_seed(hp.seed)

    # ideally, if GPU training is required, and if cuda is not available, we can raise an exception
    # however, as we want this algorithm to work locally as well (and most users don't have a GPU locally), we will fallback to using a CPU
    use_cuda = torch.cuda.is_available()
    print(f"Use cuda {use_cuda}")
    device = torch.device("cuda" if use_cuda else "cpu")

    # load data
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    training_data_loader = mnist_dataloader(hp.batch_size, train=True, **kwargs)
    test_data_loader = mnist_dataloader(hp.batch_size, train=False, **kwargs)

    # train the model
    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=hp.learning_rate, momentum=hp.sgd_momentum)

    # run multiple epochs and capture the accuracies for each epoch
    # train the model: run multiple epochs and capture the accuracies for each epoch
    accuracies = []
    for epoch in range(1, hp.epochs + 1):
        train(model, device, training_data_loader, optimizer, epoch, hp.log_interval)
        accuracies.append(test(model, device, test_data_loader))

    # after training the model, we can simply save it to disk and return it from the Flyte task as a {py:class}`flytekit.types.file.FlyteFile`
    # type, which is the ``PythonPickledFile``. ``PythonPickledFile`` is simply a decorator on the ``FlyteFile`` that records the format
    # of the serialized model as ``pickled``
    model_file = "mnist_cnn.pt"
    torch.save(model.state_dict(), model_file)

    return TrainingOutputs(epoch_accuracies=accuracies, model_state=PythonPickledFile(model_file))
    


@workflow
def pytorch_training_wf(hp: Hyperparameters = Hyperparameters(epochs=10, batch_size=128)) -> TrainingOutputs:
    return pytorch_mnist_task(hp=hp)


if __name__ == "__main__":
    model, accuracies = pytorch_training_wf(hp=Hyperparameters(epochs=10, batch_size=128))
    print(f"Model: {model}, Accuracies: {accuracies}")