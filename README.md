# CoTA: Compression Transformer Autoencoder

CoTA is a new way to represent sequential data.
Unlike the previous models, that used single vector to represent the sequential data,
CoTA uses varying number of embedding vectors.
This leads to a better preservation of details and order.

<table>
	<tr>
		<td><img src="readme/images/transformer_encoder.png" alt="transformer_encoder"/></td>
		<td><img src="readme/images/cot_encoder.png" alt="cot_encoder"/></td>
	</tr>
	<tr>
		<td style="text-align: center">Usual transformer encoder</td>
		<td style="text-align: center">CoTA encoder</td>
	</tr>
</table>

Unlike usual transformer encoders, CoTA encoder decides which vectors to keep after applying attention mechanism.
The model has been trained to accurately reconstruct the sequence while keeping the number of embeddings small.

CoTA can be adopted to any Transformer encoder model without significant effort.
This approach do not require any additional parameters to be introduced.
Zero element of the vectors after attention mechanism is used to determine whether to keep or eliminate the vector.
Vector elimination may be implemented either using attention mask or removing vectors from tensor. 

<img src="readme/images/cot_mask.png" alt="cot_mask"/>
