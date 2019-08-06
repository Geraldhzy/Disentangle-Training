# Disentangled-Training

This repo provides the code to train disentanglement for product title embedding using variational dropout. Training data should contain original title embeddings, image features and the mapping between titles and images, training output is a transforming matrix W that can transform original title embeddings to disentangled title embeddings.

## Environment
 - numpy==1.16.4
 - tensorflow==1.13.1
 - sklearn==0.21.2
 
## Data
 - Title image pairs: text format
 - Title embeddings: any vector format
 - Image features: any vector format
 
## Training

 - Weight of orthogonal loss
 
 Bigger weight requires stronger orthogonal constraint, if this loss converges to very small value, then the performances after transformation would be similar to original embeddings. So for improvements on performances, small weight is adopted in the training.
 
 - Weight of KL divergence
 
 Bigger weight encourages more sparse networks, which means more dimensions have very high dropout rates.So a very small weight is adopted here to prevent high dropout rates for Image-Informative, meanwhile manual constraints on specific dimensions assure high dropout rates for Image-Uninformative.
 
 - Other details
 
 Applying normalization on image features and original title embeddings might help to avoid much tuning on hyperparameters when training on different dataset.
 
 When to stop training? When the prediction loss is converging and the orthogonal loss is small enough, or just look at the validation accuracy.  
 
## Output

 - Transforming matrix W
 - Learned dropout rates

## Evaluation

To evaluate the learning result, firstly using W to transform original embeddings to disentangled embedding, then perform evaluations. Test data containing 2500 labelled similar title pairs can be used to test identical product classification. Searching for nearest neighbors is also a way to look into what are learned in the embeddings.
