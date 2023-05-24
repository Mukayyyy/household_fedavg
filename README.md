# FedAVG for household characteristic prediction

This code is for household characteristic prediction. The main code is in the  `lib`. 

To directly run the non-DP version, set the parameter `args.model` to the corresponding model option, and run this following command:
`cd exps`
`python feaderated_main.py`

To run the DP version, directly run:
`python feaderated_dp.py`

To record the running log, try command:
`python federated_dp.py > example_run.log`


# Running environment requirement
`requirement.txt`



## how to change

there are seveal parts in this code.
`exps/feaderated_main.py` is the main code. It defines model aggregation, the selection of different heterogeneous type. It is easy to understand.
`lib/models/*` define different model.
`lib/options.py` defines different parameter. `feaderated_main.py` will read this file and load those settings. Acually, for easy settings, I defines a series of settings between 132-143 lines in `feaderated_main.py`, just like label files and heterogeneous type. You can find them in this code.
`lib/utils.py`defines different dataloader.
`lib/update.py` defines model training part.

---

---


