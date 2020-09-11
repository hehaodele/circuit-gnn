# Circuit-GNN: Graph Neural Networks for Distributed Circuit Design
Guo Zhang*, Hao He*, Dina Katabi

(* indicates co-primary authors)

Check the [ICML proceeding](http://proceedings.mlr.press/v97/zhang19e.html) here. 
The [oral presentation](https://slideslive.com/38917395/applications) can be found here (starting from 1:09:40).

### Circuit Generator
Use `circuit.py` to randomly generating parameterized square-resonator filters.
The following command, will generated and visualize a `n`-resonator circuits.
```python
python circuit.py --num_resonator {n}
```

### Download Dataset
You can download datast here, [Circuit-GNN Dataset (27.6GB)](http://circuit-gnn.csail.mit.edu/data.zip).

Extract the zip file and put them into your `data_root`. You should find a folder `data` and three data lists `train/valid/test_list.txt` under your `data_root`.
In the `data` folder there should be multiple subfolders with the name `num{n}_type{t}`. It contains all random generated circuits having `n` resonators and topology type `t`.
The total dataset have a size of 30GB.
The dataset is an updated version of the dataset we used for the original ICML paper.
- Larger Size: It has more samples of 4/5-resonator circuits for training and validation.
- More Balanced: It has a more balanced data amount for the circuits with different topology types.
- Harder Examples: The range a circuit parameters is adjusted a bit to include circuit with unusual s2p.      

### Download Pre-trained Checkpoint
We provide an pre-trained model, [Circuit-GNN Example](http://circuit-gnn.csail.mit.edu/exp_example.zip).
After extracting it, please move it the `./dump` which is the default folder where all experiments folder will be put.
Then you can test, do forward prediction or inverse optimization with this pre-trained model via the following commands.   


### Train Circuit GNN
Start the training by the following command.
```python
python train.py --exp {your_exp_name} --data_root {your_data_root}
```
To customize training hyperparameters such as learning rate, batch size, please see the configuration file `config.py`.
The training result including model checkpoints and training logs will be stored in `./dump/{your_exp_name}`.

### Test Circuit GNN
```python
python test.py --exp {your_exp_name} --data_root {your_data_root} --epoch {model_checkpoint_epoch}
```
Following is an example result of Circuit-GNN.     
  
```bash
+----------------+----------+-----------------+------------------+------------------+-----------------+
| # of resonator | topology |   # of samples  | train error (db) | valid error (db) | test error (db) |
+----------------+----------+-----------------+------------------+------------------+-----------------+
|       4        |    0     | 12000 1500 1500 |      0.670       |      2.085       |      2.062      |
|       4        |    1     | 12000 1500 1500 |      0.685       |      2.325       |      2.329      |
|       4        |    2     | 12000 1500 1500 |      0.579       |      1.422       |      1.468      |
|       4        |    3     | 12000 1500 1500 |      0.569       |      1.419       |      1.364      |
|       4        |    4     | 12000 1500 1500 |      0.534       |      1.294       |      1.279      |
|       4        |    5     | 12000 1500 1500 |      0.561       |      1.255       |      1.306      |
|       4        |    6     | 12000 1500 1500 |      0.683       |      2.037       |      2.029      |
|       4        |    7     | 12000 1500 1500 |      0.628       |      1.592       |      1.597      |
|       4        |    8     | 12000 1500 1500 |      0.670       |      1.706       |      1.694      |
|       4        |    9     | 12000 1500 1500 |      0.655       |      1.755       |      1.734      |
|       4        |   avg    |        -        |      0.623       |      1.689       |      1.686      |
|       5        |    0     | 20000 2500 2500 |      0.713       |      2.044       |      2.101      |
|       5        |    1     | 20000 2500 2500 |      0.705       |      1.871       |      1.864      |
|       5        |    2     | 20000 2500 2500 |      0.715       |      1.852       |      1.815      |
|       5        |    3     | 20000 2500 2500 |      0.710       |      1.857       |      1.823      |
|       5        |    4     | 20000 2500 2500 |      0.673       |      1.925       |      1.899      |
|       5        |    5     | 20000 2500 2500 |      0.732       |      2.173       |      2.137      |
|       5        |    6     | 20000 2500 2500 |      0.769       |      2.510       |      2.525      |
|       5        |    7     | 20000 2500 2500 |      0.766       |      2.499       |      2.494      |
|       5        |    8     | 20000 2500 2500 |      0.778       |      2.815       |      2.829      |
|       5        |   avg    |        -        |      0.729       |      2.172       |      2.165      |
|       3        |    0     |     0 0 1000    |        -         |        -         |      1.490      |
|       3        |    1     |     0 0 1000    |        -         |        -         |      1.362      |
|       3        |    2     |     0 0 1000    |        -         |        -         |      1.260      |
|       3        |    3     |     0 0 1000    |        -         |        -         |      3.298      |
|       3        |   avg    |        -        |        -         |        -         |      1.853      |
|       6        |    0     |     0 0 900     |        -         |        -         |      4.073      |
|       6        |    1     |     0 0 900     |        -         |        -         |      3.792      |
|       6        |    2     |     0 0 900     |        -         |        -         |      4.130      |
|       6        |    3     |     0 0 900     |        -         |        -         |      4.103      |
|       6        |    4     |     0 0 900     |        -         |        -         |      2.646      |
|       6        |    5     |     0 0 900     |        -         |        -         |      3.391      |
|       6        |   avg    |        -        |        -         |        -         |      3.689      |
+----------------+----------+-----------------+------------------+------------------+-----------------+
```
Note that the error is a bit worse than the result in the original paper. It is due to that there are more hard examples (circuits with weired s2p functions) in this updated dataset.
In the original dataset, we included more easier data sample. For example, 4-resonator circuits with topology type 2.  Based on human knowledge, this topology is known to deliver more regular s2p functions like filters. 
We were biasing our model at that time since we were focus on using the model to design/optimize filters.

### Forward Prediction with Circuit GNN
To visualize the forward prediction of Circuit GNN on `n`-resonator circuits with topology type `tp`, one can just run the following command. 
Choose `phase` from `train | valid | test` to visualize result at train/validation/test dataset.  
```python
python vis_forward.py --data_root {your_data_root} --exp {your_exp_name} --epoch {checkpoint epoch} --num_resonator {n} --circuit_type {tp} --phase {phase} 
```

### Inverse Optimization with Circuit GNN
Simply run the following command, you will see an example of how the model generating a filter having a passband from `260 GHz` to `290 GHz`.
```python
python inverse.py --exp {your_exp_name} --epoch {checkpoint epoch} 
```

### Citing Our Paper
Finding our dataset and code base useful? Please consider citing:
```tex
@inproceedings{he2019circuit,
  title={Circuit-GNN: Graph neural networks for distributed circuit design},
  author={Zhang, Guo and He, Hao and Katabi, Dina},
  booktitle={International Conference on Machine Learning},
  pages={7364--7373},
  year={2019}
}
```
