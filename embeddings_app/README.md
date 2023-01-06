Embeddings App
==============

**With this app you can explore your data: find out clusters of similar objects, outliers, or wrong labels of the objects just by looking once at the chart.**


## Features
- Calculate embeddings (feature vectors) with recent pre-trained models from HuggingFace and [timm](https://huggingface.co/docs/timm/index) (such as CLIP, ConvNeXt, BEiT and others).
- Embeddings can be gathered either for every object in a dataset (cropping will be made automatically), or for images, or for both images and objects.
- Visualize your embeddins in the 2D space with a projection methods like UMAP, PCA, t-SNE or their combinations.
- The app finds changes in datasets and only recalculates the outdated embeddings (only for images that have been updated).


## How it works:
1. You select a pre-trained model to infer or input a model_name from [timm](https://huggingface.co/models?sort=downloads&search=timm%2F).
2. The app infer the model on objects and images in your dataset collecting the embeddings - outputs of model before any pooling or classification head.
3. Then embeddings will be decomposed and projected onto 2D space with the one of the projection_method: UMAP, t-SNE, PCA.
4. Now you can explore your data, clicking on points in the chart and watching the images and annotations.

**Note:**
The embeddings are calculated with large pre-trained models such as OpenAI CLIP, Facebook ConvNeXt.
These models can retrieve very complex relationships in the data, so the data samples were arranged in space by some semantic meaning, not just by color and shape of the object.
