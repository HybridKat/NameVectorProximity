import os
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(2)

################################### Préparation des données #####################
# Charge les noms dans une liste avec des marqueurs de fin(^)
listNoms = []
with open("first-names.txt", "r") as f:
    for line in f:
        listNoms.append(line.strip()+"^")
        
# Crée une liste de tous les caractères uniques dans les noms et fait deux dictionnaires
vocab = sorted(set("".join(listNoms)))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Convertit les noms en index
listNomsAsInt = []
for nom in listNoms:
    listNomsAsInt.append([char2idx[c] for c in nom])


############################## Création du modèle VAE ##############################
batchSize = 64
validationSplit = 0.2
nbEpochs = 32
latentDim = 64
retrain = True

vocabSize = len(vocab)
seqLength = max([len(nom) for nom in listNomsAsInt])

# Prépare le dataset pour l'entraînement
listNomsAsInt = keras.preprocessing.sequence.pad_sequences(listNomsAsInt, padding="post")
nomsDataset = tf.data.Dataset.from_tensor_slices(listNomsAsInt)
nomsDataset = nomsDataset.shuffle(10000, reshuffle_each_iteration=True)

# Split le dataset en train et test
nomsDatasetTrain = nomsDataset.skip(int(len(listNomsAsInt)*validationSplit)).batch(batchSize, drop_remainder=True)
nomsDatasetTest = nomsDataset.take(int(len(listNomsAsInt)*validationSplit)).batch(batchSize, drop_remainder=True)

# Classe du modèle VAE
class VAE(keras.Model):
    def __init__(self, encodeur, decodeur, **kwargs):
        super().__init__(**kwargs)
        self.encodeur = encodeur
        self.decodeur = decodeur
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
         
    @property
    def metrics(self):
        return [self.loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]
    
    def print(self, data, reconstruction):
        print("Input: ", "".join([idx2char[i] for i in data.numpy()[0]]))
        print("Reconstruction: ", "".join([idx2char[i] for i in reconstruction.numpy()[0]]))
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encodeur(data)
            reconstruction = self.decodeur(z)
            reconstruction_loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(data, reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1) * 0.01
            loss = reconstruction_loss + kl_loss

            # 0.1% de chance de print la reconstruction
            if np.random.rand() < 0.001:
                self.print(data, reconstruction)
            
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
         
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
         
        return {"loss": self.loss_tracker.result(), "reconstruction_loss": self.reconstruction_loss_tracker.result(), "kl_loss": self.kl_loss_tracker.result()}
    
    def test_step(self, data):
        z_mean, z_log_var, z = self.encodeur(data)
        reconstruction = self.decodeur(z)
        reconstruction_loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(data, reconstruction))
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1) * 0.01
        loss = reconstruction_loss + kl_loss
        
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {"loss": self.loss_tracker.result(), "reconstruction_loss": self.reconstruction_loss_tracker.result(), "kl_loss": self.kl_loss_tracker.result()}
    
    def call(self, data):
        return self.decodeur(self.encodeur(data)[2])
        
        
        
# Encodeur
encodeur = keras.Sequential([
    keras.layers.InputLayer(input_shape=[seqLength]),
    keras.layers.Embedding(vocabSize, 64),
    keras.layers.LSTM(latentDim),
    keras.layers.Dense(latentDim, activation="relu"),
])

# Ajoute z_mean et z_log_var pour le calcul de la loss
z_mean = keras.layers.Dense(latentDim, name="z_mean")(encodeur.output)
z_log_var = keras.layers.Dense(latentDim, name="z_log_var")(encodeur.output)
z = keras.layers.Lambda(lambda x: x[0] + tf.exp(0.5 * x[1]) * tf.random.normal(tf.shape(x[0])), name="z")([z_mean, z_log_var])
encodeur = keras.Model(inputs=encodeur.input, outputs=[z_mean, z_log_var, z], name="encodeur")

 
# Décodeur
decodeur = keras.Sequential([
    keras.layers.InputLayer(input_shape=[latentDim]),
    keras.layers.RepeatVector(seqLength),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(vocabSize, activation="softmax"))
])
 
# Modèle VAE
if retrain:
    earlyStopping = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    vae = VAE(encodeur, decodeur)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(nomsDatasetTrain, 
            epochs=nbEpochs, 
            steps_per_epoch=len(listNomsAsInt)*(1 - validationSplit) // batchSize, 
            validation_data=nomsDatasetTest, 
            validation_steps=len(listNomsAsInt)*validationSplit // batchSize,
            #callbacks=[earlyStopping]
    )
    # Sauvegarde l'encodeur et le décodeur
    encodeur.save_weights("vae_encodeur.h5")
    decodeur.save_weights("vae_decodeur.h5")
else:
    # Charge l'encodeur et le décodeur
    encodeur.load_weights("vae_encodeur.h5")
    decodeur.load_weights("vae_decodeur.h5")
    vae = VAE(encodeur, decodeur)
 
################################# Test sur des noms ###################################
testNames = ["James", "Marie", "Marie Anne", "Marianne", "Sheamus", "Kevin", "Morie", "Morin"]
namesEncoded = []

# Encode et décode les noms de test
for name in testNames:
    nameAsInt = [char2idx[c] for c in name]
    nameAsInt = keras.preprocessing.sequence.pad_sequences([nameAsInt], maxlen=seqLength, padding="post")
    nameAsInt = tf.convert_to_tensor(nameAsInt)
    
    z_mean, z_log_var, z = vae.encodeur.predict(nameAsInt, verbose=0)
    namesEncoded.append(z)
    decoded = vae.decodeur.predict(z, verbose=0)[0]
    
    # Reconverti les noms en string en arretant sur le caractère de fin
    decoded = [idx2char[np.argmax(c)] for c in decoded]
    decoded = "".join(decoded)
    decoded = decoded.split("^")[0]
     
    print("Nom:", name)
    print("Nom reconstruit:", decoded)
    print()
    
# Pour chaque combinaisons de noms, calcule la distance entre les deux noms
for i in range(len(testNames)):
    for j in range(i+1, len(testNames)):
        print("Distance entre", testNames[i], " et ", testNames[j], ":", np.linalg.norm(namesEncoded[i] - namesEncoded[j]))
    print()
     