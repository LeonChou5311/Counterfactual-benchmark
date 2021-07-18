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


## Can we use the `evaluate_counterfactuals` in LORE to evaludate other counterfactual algorithms?

Short answer: `NO`

To explain reason, I have to:
- Review LIME and LINDA
- What's LORE?
- Why LORE can be used for generating counterfactuals?
- The risk of the counterfactuals from LORE
- What does the `evaluate_counterfactuals` do?
- Why we can't use `evaluate_counterfactuals` for others?
- Futherwork


### LIME and LINDA:
LIME and LINDA are simalar in terms of exaplaining strategy. They first randomly sample data points around the query instance. And use the output from the black box as the label to training an interpretable model. LIME uses decision tree or other linear models. And, LINDA use bayesian network for explaining. Basically these two XAI algorithms consist of three phases:

1. Permutation.
2. Using blackbox to predict permutations.
3. The predictions of permutations will be used as label to train an interpretable model.

### What's LORE?
LORE has the same concept as LIME and LINDA. The only difference is the permutation strategy. Instead randomly sample around the query instance, LORE use genetic algorithm to generate permutations by maximising these two fitness functinos:
![image](https://user-images.githubusercontent.com/37566901/126080882-c083df9d-b2ab-4c64-8a86-95e42e5a9429.png)

||LIME|LINDA|LORE
|---|---|---|---|
|Permutation|Random|Random|Genetic Alg|
|Interpretable Model| Decision Tree/ Linear Models| Bayesian Network| Decision Tree |


### Why LORE can be used for generating counterfactuals?

The rules can be extracted from the decision tree. LORE using decision tree as their interpretable model. Therefore, rules and counterfactuals can be found by exploring the local interpretable model.


### The risk of the counterfactuals from LORE

The interpretable model and the black box are not the same model. Therefore, they may have different output value from the same input instance. In other words, the counterfactual found on the interpretable model (Decision tree) has a chance to be a fake counterfactual on the black box. 

### What does the `evaluate_counterfactuals` do?

This function is used for checking if the counterfactual found on the interpretable model is a real counterfactual on the black box.

### Why we can't use `evaluate_counterfactuals` for others?

This function is mainly used for evaluating the **interpretable model** by analysing the counterfactuals (**Not evaluating the counterfactual itself**.). If the interpretable model is a good representation of the black box in local, the counterfauctal found in interpretable model (decision tree) should be a counterfactual in black box too. 

Therefore, this function is not used for evaluating the generated counterfactuals (The function name is confusing). 

### Futherwork

What if we apply generic algorithm as the permutation strategy for LINDA? Would it improve LINDA?








