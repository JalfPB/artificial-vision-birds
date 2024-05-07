# TensorFlow y tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Layer, Embedding, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Librerias de ayuda
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm

# Lets get name of all the subfolders 
train_folder_paths = []
for x in tqdm(os.walk('./Dataset/train')):
    train_folder_paths.append(x[0])

# Lets get all the image file paths and their corresponding labels

image_file_path = []
label = []
for folder_paths in tqdm(train_folder_paths[1:], desc = "Getting the folders: ", position = 0):
    for images in os.listdir(folder_paths):
        image_path = os.path.join(folder_paths, images)
        label_name = os.path.basename(folder_paths)
        image_file_path.append(image_path)
        label.append(label_name)

train_df = pd.DataFrame()
train_df["paths"] = image_file_path
train_df["labels"] = label
train_df.head(5)

# Lets get name of all the subfolders 
train_folder_paths = []
for x in tqdm(os.walk('./Dataset/valid')):
    train_folder_paths.append(x[0])



# Lets get all the image file paths and their corresponding labels
image_file_path = []
label = []
for folder_paths in tqdm(train_folder_paths[1:], desc = "Getting the folders: ", position = 0):
    for images in os.listdir(folder_paths):
        image_path = os.path.join(folder_paths, images)
        label_name = os.path.basename(folder_paths)
        image_file_path.append(image_path)
        label.append(label_name)

# Create Dataframe:
valid_df = pd.DataFrame()
valid_df["paths"] = image_file_path
valid_df["labels"] = label
valid_df.head(5)

# Lets get name of all the subfolders 
train_folder_paths = []
for x in tqdm(os.walk('./Dataset/test/')):
    train_folder_paths.append(x[0])

# Lets get all the image file paths and their corresponding labels
image_file_path = []
label = []
for folder_paths in tqdm(train_folder_paths[1:], desc = "Getting the folders: ", position = 0):
    for images in os.listdir(folder_paths):
        image_path = os.path.join(folder_paths, images)
        label_name = os.path.basename(folder_paths)
        image_file_path.append(image_path)
        label.append(label_name)
#         print(f"{folder_paths}/{images}")

# Create Dataframe:
test_df = pd.DataFrame()
test_df["paths"] = image_file_path
test_df["labels"] = label
test_df.head(5)

class MultiHeadedAttention(Layer):
    def __init__(self, embedding_dims, no_head):
        """
            Arguments:
                - embedding_dims : Takes the dimension of the patch embeddings i.e. the length of each patch after its converted into a vector.
                - no_head : number of heads
            
            Variables:
                - The weights of the 'query', 'key' and 'value' vectors are learnt.
                - The 'embedding_dims' should be divisible by the 'no_head'.
                  By default we take the 'embedding_dims'= 768 and 'no_head'=16.
                - The 'projections_dims' = embedding_dims // no_head, that is equal to 48. 
                  Each embedding vector is divided into 16 parts and each part is of length 48.
                - The 'combined_heads' variable is the same length as the other vectors we get it after combining the 16 heads having length = 48.
                  So as embedding dims = 768, 16*48 = 768.
                
        """
        super(MultiHeadedAttention, self).__init__(name="MultiHeadSelfAttentionBlock")
        self.embedding_dims = embedding_dims
        self.no_head = no_head
        if embedding_dims % no_head != 0:
            print(f'embedding dimensions : {embedding_dims} should be divisible by number of heads : {no_head}')
        self.query = Dense(embedding_dims)
        self.key = Dense(embedding_dims)
        self.value = Dense(embedding_dims)
        self.projections_dims = embedding_dims // no_head
        self.combined_heads = Dense(embedding_dims)
        
    def attention_block(self, query, key, value):
        """
            'score' :    query                                key
                        [1,2,4,5,6,..........,256]   x       [ 1,
                                                               2,
                                                               4,
                                                               5,
                                                               ...
                                                               256]   
                                                               
            'dim_key' : This is the dimension of the keys. 
                        It’s calculated as the last dimension of the shape of the key tensor, converted to a float32 type.
                        
            'scaled_score' : This scaling operation in attention mechanisms to prevent the dot product between large vectors from growing too large.
            
            'weights' : These are the attention weights and are calculated by applying a softmax function to the scaled scores. 
                        The softmax function ensures that all the weights sum to 1, effectively turning the scores into probabilities.
            
            'output' :  This is the final output of the attention mechanism. 
        
        """
        score = tf.matmul(query, key, transpose_b = True)
        dim_key = tf.cast(tf.shape(key)[-1],  tf.float32)
        scaled_score = score/tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights
    
    def separate_heads(self, x, batch_size):
        x = tf.reshape(
                        x, 
                        (batch_size, -1, self.no_head, self.projections_dims)
                      )
        return tf.transpose(
                                x,
                                perm=[0,2,1,3]
                            )
    
    def call(self, inputs):
        
        batch_size = tf.shape(inputs)[0]
        query_vector = self.query(inputs)
        key_vector = self.key(inputs)
        value_vector = self.value(inputs)
        
        query_vector = self.separate_heads(query_vector, batch_size)
        key_vector = self.separate_heads(key_vector, batch_size)
        value_vector = self.separate_heads(value_vector, batch_size)
        
        attention, weights = self.attention_block(query_vector, key_vector, value_vector)
        attention = tf.transpose(attention, perm=[0,2,1,3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embedding_dims))
        
#         output = self.combined_heads(self.embedding_dims)
        output = self.combined_heads(inputs=concat_attention)
        
        return output
    
class TransformerBlock(Layer):
    
    def __init__(self, no_head, embedding_dims, mlp_dim, dropout):
        super(TransformerBlock,self).__init__(name="TransformerBlock")
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.mul_attention = MultiHeadedAttention(embedding_dims , no_head)
        self.mlp = tf.keras.Sequential(
            [
                Dense(mlp_dim, activation = tf.keras.activations.gelu),
                Dropout(dropout),
                Dense(embedding_dims),
                Dropout(dropout)
            ]
        )
        
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        
    def call(self, inputs):
        
        inputs_norm = self.layernorm1(inputs)
        attention_output = self.mul_attention(inputs_norm)
        attention_output = self.dropout1(attention_output)
        out = attention_output + inputs
        out_norm = self.layernorm2(out)
        mlp_output = self.mlp(out_norm)
        mlp_output = self.dropout2(mlp_output)
        
        return mlp_output + out
    
class VisionTransformer(tf.keras.Model):
    def __init__(
                    self,
                    image_size,
                    patch_size,
                    embedding_dims,
                    no_head,
                    mlp_dim,
                    no_layers,
                    dropout,
                    num_class,
                    channels
                ):
        
        super(VisionTransformer, self).__init__(name="VisionTransformer")
        
        self.num_patches = (image_size//patch_size)**2
        self.patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        self.embedding_dims = embedding_dims
        self.no_layers = no_layers
        self.rescale = tf.keras.layers.Rescaling(scale=1./255)
        self.pos_emb = self.add_weight(
            name="pos_emb", shape=(1, self.num_patches + 1, embedding_dims)
        )
        self.class_emb = self.add_weight(name="class_emb", shape=(1, 1, embedding_dims))
        self.patch_proj = Dense(embedding_dims)
        self.enc_layers = [
                                TransformerBlock(no_head, embedding_dims, mlp_dim, dropout)
                                for _ in range(no_layers)
                            ]
        self.mlp_head = tf.keras.Sequential(
                                                [
                                                    tf.keras.layers.LayerNormalization(epsilon=1e-6),
                                                    Dense(mlp_dim, activation = tf.keras.activations.gelu),
                                                    Dropout(dropout),
                                                    tf.keras.layers.Flatten(),
                                                    Dense(num_class)
                                                ]
        
        
                                            )
        
    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches
        
    def call(self, x):
        
        batch_size = tf.shape(x)[0]
        x = self.rescale(x)
        patches = self.extract_patches(x)
        x = self.patch_proj(patches)
        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.embedding_dims]
        )
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb
        

        for layer in self.enc_layers:

            x = layer(x)
        x = self.mlp_head(x)
#         x = Dense(self.num_class)(x)

        return x
    
    # Create a data generator for training data
train_datagen = ImageDataGenerator(
#     rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,  # Typically not used for mammography
    fill_mode='nearest'
    )
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='paths',
    y_col='labels',
    target_size=(224, 224),  # adjust this to the size of your images
    batch_size=8,
    class_mode='categorical'  # use 'categorical' for multi-class problems
)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model =  VisionTransformer(
                                image_size = 224,
                                patch_size = 16,
                                embedding_dims = 768,
                                no_head = 12,
                                mlp_dim = 3072,
                                no_layers = 12,
                                dropout = 0.1,
                                num_class = 525,
                                channels=3
                            )

    model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(
                    from_logits=True
                ),
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate = 1e-6
                ),
                metrics=["accuracy"],
            )
    # history = model.fit(X_train, y_train, batch_size=16, epochs=100)
    history = model.fit(train_generator, epochs=5, validation_data=valid_generator)