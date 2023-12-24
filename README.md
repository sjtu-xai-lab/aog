# Defining and Quantifying the Emergence of Sparse Concepts in DNNs

PyTorch Implementation of the paper "Defining and Quantifying the Emergence of Sparse Concepts in DNNs" (CVPR 2023) [arxiv](https://arxiv.org/abs/2111.06206)

## Requirements

~~~bash
pip install -r requirements.txt
~~~

## Minimal Examples

- To compute interactions

  ~~~python
  from harsanyi import AndHarsanyi
  
  # model: Callable = ... (the model, i.e. a callable function)
  # selected_dim: str = ... (the output dimension of the model to be explained. Please see interaction_utils.py)
  # sample: torch.Tensor = ... (the input sample)
  # baseline: torch.Tensor = ... (the baseline value)
  # label: Union[torch.Tensor, int] = ... (the ground-truth label)
  # all_players: List = ... (a list indicating all players, len=n)
  # mask_input_fn: Callable = ... ([sample, baseline, [S1, S2, ...]] -> masked samples [x_S1, x_S2, ...])
  # verbose: int = ... (print intermediate progress)
  
  calculator = AndHarsanyi(
      model         = model,
      selected_dim  = selected_dim,
      x             = sample,
      baseline      = baseline_init,
      y             = label,
      all_players   = all_players,
      mask_input_fn = mask_input_fn,
      verbose       = verbose
  )
  calculator.attribute()
  
  interactions = calculator.get_interaction()  # [2^n,]   interaction for each S
  masks = calculator.get_masks()               # [2^n, n] bool mask for each S
  ~~~

- To learn the optimal baseline value

  ~~~python
  from harsanyi import AndBaselineSparsifier
  
  # calculator: AndHarsanyi = ... (defined above)
  # loss: str = ... (the loss to optimize baseline value, in this paper Lasso)
  # baseline_min, baseline_max: torch.Tensor = ... (the lower/upper bound for the baseline value)
  # baseline_lr: float = ... (learning rate)
  # niter: int = ... (how many iterations to optimize the baseline value)
  
  sparsifer = AndBaselineSparsifier(
      calculator   = calculator, 
      loss         = "l1",
      baseline_min = baseline_min,
      baseline_max = baseline_max,
      baseline_lr  = 1e-3, 
      niter        = 50
  )
  sparsifer.sparsify()
  
  interactions = sparsifer.get_interaction()
  masks = sparsifer.get_masks()
  ~~~

- To visualize the AOG

  ~~~python
  from harsanyi.remove_noisy import remove_noisy_greedy
  from tools.aggregate import aggregate_pattern_iterative
  from graph.aog_utils import construct_AOG
  
  # 1. remove noisy patterns
  selected_indices = remove_noisy_greedy(
      rewards, interactions.clone(), masks
  )["final_retained"]
  
  #  2. aggregate common coalitions
  merged_patterns, aggregated_concepts, _ = aggregate_pattern_iterative(
      patterns     = masks[selected_indices],
      interactions = interactions[selected_indices],
  )
  single_features = np.vstack([np.eye(len(words)).astype(bool), merged_patterns])
  
  # 3. construct AOG
  aog = construct_AOG(
      attributes      = words,
      single_features = single_features,
      concepts        = aggregated_concepts,
      interactions    = interactions[selected_indices]
  )
  
  # 4. construct AOG
  aog.visualize(
      figsize        = (15, 7),
      save_path      = "aog.svg", # or "aog.html" for an interactive demo 
      renderer       = "networkx",
      highlight_path = "rank-1",
      title          = "dummy title"
  )
  ~~~

## Usage

- [x] models: tabular, NLP, image
- [x] datasets: tabular (commercial, census, bike), NLP (SST-2, CoLA), image (MNIST)
- [x] demo of a whole pipeline

**Compute interactions and visualize the AOG**

The following notebook shows how to visualize an AOG based on computed interactions. [![Maintenance](https://img.shields.io/badge/Open%20in-nbviewer-orange.svg)](https://nbviewer.org/github/sjtu-xai-lab/aog/blob/main/src/demo_sentiment_classification.ipynb)

**Beforehand: train a model**

The following code shows two examples of how to train a model based on tabular/NLP datasets.

~~~bash
cd ./src

# tabular: 2-layer-MLP @ census
python3 train_model.py --data-root=/folder/of/the/dataset \
  --device=0 --dataset=census --arch=mlp2_logistic --seed=0 \
  --batch_size=512 --lr=0.01 --logspace=1 --epoch=500

# NLP: LSTM @ SST-2
python3 train_model.py --data-root=/folder/of/the/dataset \
  --device=0 --dataset=sst2 --arch=lstm2_uni --seed=0 \
  --batch-size=64 --lr=0.001 --logspace=1 --n-epoch=200
~~~

## Demos

Here are some demos which reproduce experimental results in this paper. You can also clone this repo and try the `ipynb` files on your own.

- A demo for an sentiment classification example [![Maintenance](https://img.shields.io/badge/Open%20in-nbviewer-orange.svg)](https://nbviewer.org/github/sjtu-xai-lab/aog/blob/main/src/demo_sentiment_classification.ipynb)
- *(Under construction, stay tuned ...)*

<img src=".\images\aog_demo.gif"></img>

## Citation

~~~latex
@InProceedings{Ren_2023_CVPR,
    author    = {Ren, Jie and Li, Mingjie and Chen, Qirui and Deng, Huiqi and Zhang, Quanshi},
    title     = {Defining and Quantifying the Emergence of Sparse Concepts in DNNs},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {20280-20289}
}
~~~

