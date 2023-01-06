Embeddings App
==============

With this app you can explore your data, find out outliers or clusters of
similar objects and more just by looking once at the chart.
 

The app calculates and visualizes embeddings (or feature vectors) for images
and objects in a dataset. The embeddings are calculated with pre-trained
models like OpenAI CLIP, Facebook Convnext and other recent models. These models
can retrieve very complex relationships in the data, arranging data points
by some semantic meaning, not just by color and shape of the object. It's
possible to use any model from [timm](https://huggingface.co/docs/timm/index)
package, just provide a [timm's
model_name](https://huggingface.co/models?sort=downloads&search=timm%2F) in the
field for that.
 

The app visualizes your data with the embedding projections into 2D space. As
for the projection method, UMAP, PCA, t-SNE and their combinations can be used.
