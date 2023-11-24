# Dataset Download

Please download the Fashion-MNIST dataset to the following path: ```"../../fashion-mnist-master"``` and the CIFAR-10 dataset to this path: ```"../../CIFAR10"```. You can adjust the root setting in either `CIFAR10.py` or `MINST.py` to match your dataset location.

Recommended Dependencies: Please check the `requirements.txt`. Our system has been tested using these settings.

# Code Running

For the evaluation of target model poisoning, the federated learning simulation program is `main1.py`, while for other attack methods it is `main.py`.

To run the code without any customized settings, please use:

```bash
python main.py
```
or
```bash
python main1.py
```

However, if you want to run the code successfully with your customized parameters, you will also need to set the following basic hyper-parameters:

| Parameter | Function                                                     |
|-----------|--------------------------------------------------------------|
| -cn       | Number of clients                                            |
| -mn       | Number of malicious clients (It can be a list)               |
| -cr       | Total number of communication rounds                         |
| -ce       | Number of training epochs for each client                    |
| -nid      | Non-IID degree (It can be a list)                            |
| --id      | Use IID dataset (It will override -nid)                      |
| -attack   | Number of attack types (It can be a list, only in `main.py`) |
| -defence  | Number of defence types (It can be a list)                   |
| -dataset  | Dataset used for training (can only be 'MINST' or 'CIFAR10') |
| -Cs       | Threshold $C_s$ used by the server                           |
| -Cc       | Threshold $C_s$ used by the client                           |
| -Cp       | Threshold to trigger the penalty mechanism                   |
| -Ca       | The award value used by the award mechanism                  |
| -rs       | The size of the root test dataset                            |
| -device   | The device to run code                                       |

The mapping between attack types and numbers in `main.py` is as follows:

| Number   | Attack type         |
|----------|---------------------|
| 0        | No attack           |
| 1        | Sign flipping       |
| 2        | Label flipping      |

The mapping between defence types and numbers in both `.py` files is as follows:

| Number | Defence type           |
|--------|------------------------|
| 0      | No defence             |
| 1      | Krum                   |
| 2      | Coordinate wise Median |
| 3      | FLTrust                |
| 4      | Siren                  |
| 5      | Siren_fl               |
| 6      | Siren_PA               |

If you use more than one value in a parameter that can be a list, the program will iterate through each combination of these values.

For more parameters and details on the above parameters, you can modify the `.py` files directly.

If you want to use the same settings as us, here are some examples:

```bash
python main1.py -device 'cpu' -Cs 9 -Cc 3 -Cp 4.5 -Ca 0.5 -rs 100 -cn 10 -mn 3 5  -cr 20 -ce 3 -nid 0.3 0.5 -defence 0  6 -dataset 'CIFAR10' 
```

```bash
python main.py -Cs 10 -Cc 4 -Cp 4.5 -Ca 0.5 -rs 100 -cn 10 -mn 4 8 -cr 5 -ce 1 -nid 0.3 0.1 -attack 0 2  -defence 0  6 -dataset 'CIFAR10' 
```

# Result Reading

After running the code, the results will be recorded in the same directory as the `.py` file. `mlog.txt` or `tlog.txt` will record the final simulation results, while the `.pth` file is a dictionary that records the results of every communication round. Its key is of type `str` and records the experimental condition, while the value is of type `torch.tensor` and records accuracy in each communication round. For more information about the output file, you can refer to the source code.