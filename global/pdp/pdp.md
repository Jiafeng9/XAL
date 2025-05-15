# Partial Dependence Plot (PDP)

Partial Dependence Plot (PDP) is a **model-agnostic global interpretability method** that visualizes the **average effect** of one or two features on the predicted outcome of a machine learning model.

It helps answer questions like:
> *"On average, as this feature increases, how does the model prediction change?"*

---

## Goal

Given a feature (or two), show how changing its value affects the model's output, while averaging out the effects of all other features.

---

##  Mathematical Definition

Let:

- $ \hat{f} $ be the trained prediction function,
- $ x_S $ be the set of features of interest (1 or 2),
- $ X_C $ be the remaining features (the complement of $ S $).

Then the **partial dependence function** is defined as:

$$
\hat{f}_S(x_S) = \mathbb{E}_{X_C}[\hat{f}(x_S, X_C)] = \int \hat{f}(x_S, x_C) \, dP(x_C)
$$

## Practical Estimation

In practice, we do **not** compute the expectation over all possible combinations of the other features $X_C$ (which would be exponentially large).

Instead, we **approximate** the expectation by using the actual samples in the training dataset:

> We fix the value of the target feature $x_S = v$, and for each training sample, keep the remaining features $x_C^{(i)}$ unchanged.  
> We then compute the model prediction on this modified input, and average over all samples.

This corresponds to the empirical approximation:

$$
\hat{f}_S(v) = \frac{1}{n} \sum_{i=1}^{n} \hat{f}(v, x_C^{(i)})
$$

This way, the estimate reflects the real-world distribution of the data and avoids generating unrealistic feature combinations.


### Parameter Definitions:

| Symbol             | Meaning                                                  |
|--------------------|----------------------------------------------------------|
| $ \hat{f} $       | Trained model (e.g., neural network, random forest)      |
| $ x_S $          | Value(s) of the feature(s) we want to study              |
| $ X_C $           | The other features not in $ S $                        |
| $ x_C^{(i)} $     | The value of $ X_C $ in the $ i $-th data sample     |
| $ n $             | Number of samples in the dataset                         |
| $ \mathbb{E}_{X_C} $ | Expectation over the joint distribution of $ X_C $ |

---

## 🛠 Algorithm Steps (Single Feature)

1. **Select** the feature of interest $ x_S $.
2. **Generate** a grid of values across the feature's domain (e.g., 10–100 in steps of 10).
3. **For each value $ v $ in the grid**:
    - Replace the feature column in all samples with $ v $, leaving other features unchanged.
    - Pass these modified samples into the model to get predictions.
    - Average the predictions to get $ \hat{f}_S(v) $.
4. **Plot** $ \hat{f}_S(v) $ vs $ v $ — this is the PDP curve.

---

## Visualization

- **X-axis**: The chosen feature value.
- **Y-axis**: The model's average predicted output at that value.
- **Plot**: The PDP curve shows the marginal effect of the feature.

---

## Advantages

- **Model-agnostic**: Works for any black-box model.
- **Global insight**: Shows the general effect of the feature across the dataset.
- **Easy to visualize**: 1D/2D PDP plots are intuitive and interpretable.

---

## Limitations

### 1. **Feature Independence Assumption**
PDP assumes the feature of interest is **independent** from the other features.  
If features are highly correlated, replacing one feature value while keeping others fixed may create unrealistic data points → misleading results.

### 2. **Average Overlap**
PDP averages the effect over the entire dataset. If the relationship between feature and output depends on specific interactions, PDP may hide it.

### 3. **Data Distribution Blindness**
PDP may include values that are rare or even never appear in the training data. These "out-of-distribution" values can lead to unreliable model predictions.

---

