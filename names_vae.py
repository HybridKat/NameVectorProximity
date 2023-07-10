import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

tf.config.run_functions_eagerly(True)

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
nbEpochs = 10
latentDim = 32
retrain = True

vocabSize = len(vocab)
seqLength = max([len(nom) for nom in listNomsAsInt])

# Prépare le dataset pour l'entraînement
listNomsAsInt = keras.preprocessing.sequence.pad_sequences(listNomsAsInt, padding="post")
nomsDataset = tf.data.Dataset.from_tensor_slices(listNomsAsInt)
nomsDataset = nomsDataset.shuffle(10000)

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
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encodeur(data)
            reconstruction = self.decodeur(z)
            reconstruction_loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(data, reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
            loss = reconstruction_loss + kl_loss
            
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
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
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
    vae = VAE(encodeur, decodeur)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(nomsDatasetTrain, 
            epochs=nbEpochs, 
            steps_per_epoch=len(listNomsAsInt)*(1 - validationSplit) // batchSize, 
            validation_data=nomsDatasetTest, 
            validation_steps=len(listNomsAsInt)*validationSplit // batchSize
    )
    # Sauvegarde l'encodeur et le décodeur
    encodeur.save_weights("encodeur.h5")
    decodeur.save_weights("decodeur.h5")
else:
    # Charge l'encodeur et le décodeur
    encodeur.load_weights("encodeur.h5")
    decodeur.load_weights("decodeur.h5")
    vae = VAE(encodeur, decodeur)
 
################################# Test sur des noms ###################################
testNames = ["James", "Marie", "Marie Anne", "Marianne"]

# Convertit les noms en index
testNamesAsInt = []
for name in testNames:
    testNamesAsInt.append([char2idx[c] for c in name])
testNamesAsInt = tf.keras.preprocessing.sequence.pad_sequences(testNamesAsInt, maxlen=seqLength, padding="post")

# Encode et décode les noms
z_mean, z_log_var, z = encodeur(testNamesAsInt)
decodedNames = decodeur(z)
 
# Affiche les noms décodés (arrêter sur le caractère de fin)
for i in range(len(testNames)):
    print("Nom encodé : " + testNames[i])
    print("Nom décodé : ", end="")
    for j in range(seqLength):
        if idx2char[np.argmax(decodedNames[i][j])] == "^":
            break
        print(idx2char[np.argmax(decodedNames[i][j])], end="")
    print("\n")
    
