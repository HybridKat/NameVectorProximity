import os
import sys
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


############################## Création du modèle AutoEncoder ##############################
batchSize = 64
validationSplit = 0.2
nbEpochs = 10
latentDim = 64
randSeed = 12241
retrain = True

vocabSize = len(vocab)
seqLength = max([len(nom) for nom in listNomsAsInt])

# Prépare le dataset pour l'entraînement
listNomsAsInt = keras.preprocessing.sequence.pad_sequences(listNomsAsInt, padding="post")
nomsDataset = tf.data.Dataset.from_tensor_slices(listNomsAsInt)
nomsDataset = nomsDataset.shuffle(10000, seed=randSeed)

# Split le dataset en train et test
nomsDatasetTrain = nomsDataset.skip(int(len(listNomsAsInt)*validationSplit)).shuffle(10000, reshuffle_each_iteration=True).batch(batchSize, drop_remainder=True)
nomsDatasetTest = nomsDataset.take(int(len(listNomsAsInt)*validationSplit)).shuffle(10000, reshuffle_each_iteration=True).batch(batchSize, drop_remainder=True)

# Classe du modèle Autoencoder
class AutoEncoder(keras.Model):
    def __init__(self, encodeur, decodeur, **kwargs):
        super().__init__(**kwargs)
        self.encodeur = encodeur
        self.decodeur = decodeur
        self.loss_tracker = keras.metrics.Mean(name="loss")
         
    @property
    def metrics(self):
        return [self.loss_tracker]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstruction = self.decodeur(self.encodeur(data))
            loss = self.compiled_loss(data, reconstruction)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    def test_step(self, data):
        reconstruction = self.decodeur(self.encodeur(data))
        loss = self.compiled_loss(data, reconstruction)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    def call(self, data):
        return self.decodeur(self.encodeur(data))
        
        
        
# Encodeur
encodeur = keras.Sequential([
    keras.layers.InputLayer(input_shape=[seqLength]),
    keras.layers.Embedding(vocabSize, latentDim*4),
    keras.layers.LSTM(latentDim*2),
    keras.layers.Dense(latentDim*2, activation="relu"),
    keras.layers.Dense(latentDim, activation="relu"),
], name="encodeur")

 
# Décodeur
decodeur = keras.Sequential([
    keras.layers.InputLayer(input_shape=[latentDim]),
    keras.layers.Dense(latentDim*2, activation="relu"),
    keras.layers.RepeatVector(seqLength),
    keras.layers.LSTM(latentDim*2, return_sequences=True),
    keras.layers.Dense(vocabSize, activation="softmax")
])

# Fonction de perte
def loss_fn(y_true, y_pred):
    # Calcul de la perte de cross-entropy catégorielle
    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return loss

# Fonction de warmup
class LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_progress = global_step / warmup_steps
        warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
        return tf.cond(
            global_step < warmup_steps,
            lambda: warmup_learning_rate,
            lambda: self.post_warmup_learning_rate,
        )
nbTrainSteps = len(nomsDatasetTrain) * nbEpochs
nbWarmupSteps = nbTrainSteps // 10
lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=nbWarmupSteps)

# Callback pour afficher l'avancement des noms
class ProgressCallback(keras.callbacks.Callback):
    def __init__(self, test_name):
        super().__init__()
        self.test_name = test_name
        name_as_int = [char2idx[c] for c in test_name]
        name_as_int = keras.preprocessing.sequence.pad_sequences([name_as_int], maxlen=seqLength, padding="post")
        self.name_as_int = tf.convert_to_tensor(name_as_int)
    
    def on_train_batch_end(self, batch, logs=None):
        if batch % 10 == 0:
            reconstructed_name = self.model(self.name_as_int)
            reconstructed_name = tf.squeeze(reconstructed_name, axis=0)

            # Convertir les séquences de reconstruction en noms
            decoded_name = [idx2char[i] for i in tf.argmax(reconstructed_name, axis=1).numpy()]
            decoded_name = "".join(decoded_name).split("^")[0]

            print()
            sys.stdout.write("\033[K")
            print("Reconstruction de James:", decoded_name)
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[F")

 
# Modèle AutoEncoder
if retrain:
    earlyStopping = keras.callbacks.EarlyStopping(
        patience=5, 
        restore_best_weights=True, 
        monitor="val_loss", 
        #start_from_epoch=16
    )
    progressCallback = ProgressCallback(test_name="James")

    ae = AutoEncoder(encodeur, decodeur)
    ae.compile(optimizer=keras.optimizers.Adam(lr_schedule), loss=loss_fn)
    ae.fit(nomsDatasetTrain, 
            epochs=nbEpochs, 
            steps_per_epoch=len(listNomsAsInt)*(1 - validationSplit) // batchSize, 
            validation_data=nomsDatasetTest, 
            validation_steps=len(listNomsAsInt)*validationSplit // batchSize,
            callbacks=[progressCallback]#[progressCallback, earlyStopping]
    )
    # Sauvegarde l'encodeur et le décodeur
    encodeur.save_weights("encodeur.h5")
    decodeur.save_weights("decodeur.h5")
else:
    # Charge l'encodeur et le décodeur
    encodeur.load_weights("encodeur.h5")
    decodeur.load_weights("decodeur.h5")
    ae = AutoEncoder(encodeur, decodeur)
 
################################# Test sur des noms ###################################
testNames = ["James", "Marie", "Marie Anne", "Marianne", "Sheamus", "Kevin", "Morie", "Morin"]
namesEncoded = []

# Encode et décode les noms de test
for name in testNames:
    nameAsInt = [char2idx[c] for c in name]
    nameAsInt = keras.preprocessing.sequence.pad_sequences([nameAsInt], maxlen=seqLength, padding="post")
    nameAsInt = tf.convert_to_tensor(nameAsInt)

    reconstructed_name = ae(nameAsInt)
    reconstructed_name = tf.squeeze(reconstructed_name, axis=0)
    
    namesEncoded.append(ae.encodeur(nameAsInt))

    # Convertir les séquences de reconstruction en noms
    decoded_name = [idx2char[i] for i in tf.argmax(reconstructed_name, axis=1).numpy()]
    decoded_name = "".join(decoded_name).split("^")[0]

    print("Nom:", name)
    print("Reconstruction:", decoded_name)
    print()
    
# Pour chaque combinaisons de noms, calcule la distance entre les deux noms
for i in range(len(testNames)):
    for j in range(i+1, len(testNames)):
        print("Distance entre", testNames[i], " et ", testNames[j], ":", np.linalg.norm(namesEncoded[i] - namesEncoded[j]))
    print()