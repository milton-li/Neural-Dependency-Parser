import math
import os
import time
from datetime import datetime

import torch
from ff_model import FFModel
from cnn_model import CNNModel
from torch import nn, optim
from tqdm import tqdm
from utils.featurize import AverageMeter, load_and_preprocess_data, minibatches


def train(classifier,
          train_data,
          dev_data,
          output_path,
          batch_size=1024,
          n_epochs=10,
          lr=0.0005):
    best_dev_UAS = 0

    optimizer = optim.Adam(classifier.model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        dev_UAS = train_for_epoch(classifier, train_data, dev_data, optimizer,
                                  loss_func, batch_size)
        if dev_UAS > best_dev_UAS:
            best_dev_UAS = dev_UAS
            print("New best dev UAS! Saving model.")
            torch.save(classifier.model.state_dict(), output_path)
        print("")


def train_for_epoch(parser, train_data, dev_data, optimizer, loss_func,
                    batch_size):
    parser.model.train()
    n_minibatches = math.ceil(len(train_data) / batch_size)
    loss_meter = AverageMeter()

    with tqdm(total=(n_minibatches)) as prog:
        for i, (train_x, train_y) in enumerate(
                minibatches(train_data, batch_size)):
            optimizer.zero_grad()
            train_x = torch.from_numpy(train_x).long()
            train_y = torch.from_numpy(train_y.nonzero()[1]).long()

            logits = parser.model(train_x)
            loss = loss_func(logits, train_y)
            loss.backward()
            optimizer.step()
            prog.update(1)
            loss_meter.update(loss.item())

    print("Average Train Loss: {}".format(loss_meter.avg))

    print("Evaluating on dev set", )
    parser.model.eval()
    dev_UAS, _ = parser.parse(dev_data)
    print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
    return dev_UAS


if __name__ == "__main__":
    debug = False

    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)

    start = time.time()
    model = CNNModel(embeddings)
    parser.model = model
    print("took {:.2f} seconds\n".format(time.time() - start))

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train(
        parser,
        train_data,
        dev_data,
        output_path,
        batch_size=1024,
        n_epochs=20,
        lr=0.0005)

    if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        parser.model.load_state_dict(torch.load(output_path))
        print("Final evaluation on test set", )
        parser.model.eval()
        UAS, dependencies = parser.parse(test_data)
        print("- test UAS: {:.2f}".format(UAS * 100.0))
        print("Done!")
