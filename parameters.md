# New result after fixing UnjustifiedCF

The issue is the difference between float32 and float64, the unjustified algorithm return the counterfactual in float32 format while the input x is float64.
While comparing x (float64) and counterfactual (float32), the system just think they are different number, which causing that huge sparsity.
Solution to this is to transform our input x to float32, and everything works fine right now. The below is figure is the new result. Only sparsity, sparsity-rate and runninng time changed.

![image](https://user-images.githubusercontent.com/48231558/186355481-c0bd25d4-ba7a-44f3-aaff-ccdef945a85a.png)

# Model Parameters

## Decision Tree
We used the default instance creator from sklearn, so every parameter is default. 
```python
class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0)
```
[Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) for this initialisation function.
 
## Random Forest
We used the default instance creator from sklearn, so every parameter is default. 
``` python
class sklearn.ensemble.RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
```
[Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) for this initialisation function.
## Neuroal Network 
**Model Architecture**:
```
tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(24,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(1),
                tf.keras.layers.Activation(tf.nn.sigmoid),
            ]
        )
```
**Optimiser**: Adam

**Loss**: Cross Entropy

# Counterfactual Algorithm Parameterss

## Watcher CF

In our implementation, we restricted the feature range to let the algorithm find the counterfactuals within the range of dataset [1]. All other parameters are remained as default.

```python
CounterFactual(
            predict_fn,
            shape,
            feature_range=feature_range, # --- [1]
        )
```

### Default initialisation function
```python
classalibi.explainers.Counterfactual(predict_fn, shape, distance_fn='l1', target_proba=1.0, target_class='other', max_iter=1000, early_stop=50, lam_init=0.1, max_lam_steps=10, tol=0.05, learning_rate_init=0.1, feature_range=(- 10000000000.0, 10000000000.0), eps=0.01, init='identity', decay=True, write_dir=None, debug=False, sess=None)
```

### Loss and distance function
Loss term:
![image](https://user-images.githubusercontent.com/48231558/186340406-c124590b-d150-4db9-8da2-c80138685e7a.png)

And in our implementation, we use L1 as the distance measurement.

## Prototype CF
Same as watche, excpet the `feature_range` is set for dataset. Other parameters remain as default.

```
CounterfactualProto(
            predict,
            shape,
            feature_range=feature_range,
        )
```

### Default initialisation function
```python
classalibi.explainers.CounterfactualProto(predict, shape, kappa=0.0, beta=0.1, feature_range=(- 10000000000.0, 10000000000.0), gamma=0.0, ae_model=None, enc_model=None, theta=0.0, cat_vars=None, ohe=False, use_kdtree=False, learning_rate_init=0.01, max_iterations=1000, c_init=10.0, c_steps=10, eps=(0.001, 0.001), clip=(- 1000.0, 1000.0), update_num_grad=1, write_dir=None, sess=None)
```

### Loss and distance function 

Loss term:
![image](https://user-images.githubusercontent.com/48231558/186340924-8b65998c-3ffb-4fc3-8ce6-1d6b36a22d6b.png)

Both L1 and L2 are used in Prototype, and their weight are controlled by betta value, which is 0.1 in our implementation (default value). 


### Prototype has a long running time
In prototype, it requires special loss terms, L_{AE} and L_{proto}, which need run the autoencoders and it's computational and time-consuming.

![image](https://user-images.githubusercontent.com/48231558/186345999-91aadb6b-2808-446d-aa3c-afdced4892bf.png)
![image](https://user-images.githubusercontent.com/48231558/186346112-658311a0-44b6-49b3-b96d-c50014995ca3.png)


## DiCE
In DiCE, the `feature_range` is not provided as an argument. However, DiCE is able to infer the feature range automatically from dataset. 

```python
dice_cf.generate_counterfactuals(
        x,
        desired_class="opposite",
        verbose=True,

        ## the three parameters below is required to restrict the usage of memory, or the program will crash. 
        total_CFs=2,
        sample_size=sample_size,
        posthoc_sparsity_param=None
        )
```

### Default initialisation function
```python
generate_counterfactuals(query_instances, total_CFs, desired_class='opposite', desired_range=None, permitted_range=None, features_to_vary='all', stopping_threshold=0.5, posthoc_sparsity_param=0.1, proximity_weight=0.2, sparsity_weight=0.2, diversity_weight=5.0, categorical_penalty=0.1, posthoc_sparsity_algorithm='linear', verbose=False, **kwargs)
```

### Loss and Distance Function

Loss function:
![image](https://user-images.githubusercontent.com/48231558/186342424-8d00eac1-f8d2-47ad-b9a0-3cdf7e52629b.png)

And the distance in the loss function is actually the IMAD value:
![image](https://user-images.githubusercontent.com/48231558/186342207-5ebdd6f8-84c8-4c4a-8984-8b470c831d89.png)


## UnjustifiedCF

In unjustifiedCF, we use all the default parameters. In terms of `feature_range`, no f

```python
instance_cf = cf.CounterfactualExplanation(x, predict, method='GS')
instance_cf.fit(verbose=True)
```

### Default initialisation function
```python
fit(self, caps=None, n_in_layer=2000, first_radius=0.1, dicrease_radius=10, sparse=True, verbose=False)
```

### Loss and Distance Function

Loss function:

![image](https://user-images.githubusercontent.com/48231558/186343804-2e5e3aaa-df30-4f0f-9023-023a8785d927.png)

And, the distance they're trying to minimise is `L2 + (gamma * sparsity)`


