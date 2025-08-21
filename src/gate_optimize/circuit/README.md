### Train once and generalize: zero-shot quantum state preparation with RL

Submission to ICLR 2025.

Running instructions (tested with Python 3.11):

1. Install the required packages:
```pip install -r requirements.txt```

2. Run the main script:
```python runner.py -fromjson config.json```
where `config.json` is the JSON file containing all necessary hyper-parameters for the training. Some sample configurations are provided in the `tests` folder.

3. To generate random test cases, use
```python random_testbench.py -n <N> -just-gen <q> -name <filename>``` to generate `N` random test cases, and `q` is the number of qubits. The test cases will be saved in `filename`.

4. To benchmark a model on a test set, use
```python random_testbench.py -hyp "q,tol,name,exptdate" -n N -name <filename>```
where `hyp` is a string containing four hyper-parameters; you will find their values in `hyper-params.json` inside the subdirectory of `plots` corresponding to the model. `N` is the number of test cases to run, and `filename` is the name of the file containing the test cases.