# Counterfactual-benchmark

```
root
|-- DiCE    
|-- alibi-watcher-counterfactual
|-- alibi-counterfactual-prototype
|-- lore
|-- face
|-- truth
|-- datasets
|-- models
|-- utils // storing all common utilities

dice_test.ipynb
alibi_watcher_cf_test.ipynb
alibi_cf_proto_test.ipynb
lore_test.ipynb
face_test.ipynb
truth_test.ipynb

```
## CUDA problem

the official tutorial only show the instruction for Ubuntu 18.04. However, that one doesn't work on Ubuntu 20.04.

Solution was `sudo apt-get install cuda`, which install and set up all the dependencies for me.


