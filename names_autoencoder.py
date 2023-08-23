import math
import os
import sys
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import string
from Levenshtein import distance as lev

tf.config.run_functions_eagerly(True)
tf.autograph.set_verbosity(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('ERROR')
tf.debugging.experimental.disable_dump_debug_info()


################################ Parametres #################################
randSeed = 5785
retrain = True

if retrain:
    param_dist = {
        'latentDim': [32, 64, 128, 256],
        'latentDimInterne': [64, 128, 256, 512],
        'optimizerChoice': ['Adam', 'Nadam'],
        'learningRate': np.logspace(-5, 0, num=10, endpoint=False),
        'decaySteps': [10000, 50000, 100000, 1000000],
        'decayRate': [0.9, 0.95, 0.99],
        'dropoutRate': [0.01, 0.1, 0.25],
        'epochs': [1024],
        'batch_size': [64, 128, 256],
        'validation_split': [0.2],
    }
    # Calcul du nombre d'itérations
    nbIterations = 30
else:
    param_dist = {}

################################### Préparation des données #####################
# Charge et nettoie les noms dans une liste avec des marqueurs de fin(^)
listNoms = []
validChars = string.ascii_letters + "-'’ ÀÁÂÃÄÅàáâãäåÈÉÊËèéêëÌÍÎÏìíîïÒÓÔÕÖØòóôõöøÙÚÛÜùúûüÝýÇç"
with open("first-names.txt", "r") as f:
    for line in f:
        clean = ''.join(i for i in line if i in validChars)
        clean = clean.replace(",", " ").strip().lower()

        # Enleve les accents     
        clean = clean.replace("à", "a")
        clean = clean.replace("á", "a")
        clean = clean.replace("â", "a")
        clean = clean.replace("ã", "a")
        clean = clean.replace("ä", "a")
        clean = clean.replace("å", "a")
        clean = clean.replace("è", "e")
        clean = clean.replace("é", "e")
        clean = clean.replace("ê", "e")
        clean = clean.replace("ë", "e")
        clean = clean.replace("ì", "i")
        clean = clean.replace("í", "i")
        clean = clean.replace("î", "i")
        clean = clean.replace("ï", "i")
        clean = clean.replace("ò", "o")
        clean = clean.replace("ó", "o")
        clean = clean.replace("ô", "o")
        clean = clean.replace("õ", "o")
        clean = clean.replace("ö", "o")
        clean = clean.replace("ø", "o")
        clean = clean.replace("ù", "u")
        clean = clean.replace("ú", "u")
        clean = clean.replace("û", "u")
        clean = clean.replace("ü", "u")
        clean = clean.replace("ý", "y")
        clean = clean.replace("ç", "c")
        clean = clean.replace("’", "'")

        if clean!= "":
            listNoms.append(clean)#+"^")

# Enleve les Joseph ou Marie au début des noms si ce n'est pas le seul prénom
for i in range(len(listNoms)):
    nom = listNoms[i]
    if nom.startswith("joseph ") and len(nom.split(" ")) > 2:
        listNoms[i] = nom[7:]
    elif nom.startswith("marie ") and len(nom.split(" ")) > 2:
        listNoms[i] = nom[6:]
    elif nom.startswith("j ") and len(nom.split(" ")) > 2:
        listNoms[i] = nom[2:]
    elif nom.startswith("m ") and len(nom.split(" ")) > 2:
        listNoms[i] = nom[2:]

listNoms = list(dict.fromkeys(listNoms))
        
# Crée une liste de tous les caractères uniques dans les noms et fait deux dictionnaires
vocab = sorted(set("".join(listNoms)))
print(vocab)
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Enregistre les dictionnaires dans un fichier
with open("char2idx.txt", "w") as f:
    for key, value in char2idx.items():
        f.write(str(key)+"\t"+str(value)+"\n")

with open("idx2char.txt", "w") as f:
    for value in idx2char:
        f.write(str(value)+"\n")

# Convertit les noms en index
listNomsAsInt = []
for nom in listNoms:
    if(len(nom) <= 25):
        listNomsAsInt.append([char2idx[c] for c in nom])


############################## Création du modèle AutoEncoder ##############################
vocabSize = len(vocab)
seqLength = max([len(nom) for nom in listNomsAsInt])

# Prépare le dataset pour l'entraînement
listNomsAsInt = keras.preprocessing.sequence.pad_sequences(listNomsAsInt, padding="post")
nomsDataset =  np.array(listNomsAsInt)

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
            reconstruction = self.decodeur(self.encodeur(data[0]))
            loss = self.compiled_loss(data[0], reconstruction)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    def test_step(self, data):
        reconstruction = self.decodeur(self.encodeur(data[0]))
        loss = self.compiled_loss(data[0], reconstruction)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    def call(self, data):
        return self.decodeur(self.encodeur(data))  


# Fonction de perte
def loss_fn(y_true, y_pred):
    # Calcul de la perte de cross-entropy catégorielle
    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    # Pour chaque nom, calcule la distance de Levenshtein entre le nom original et le nom reconstruit
    inputs = y_true.numpy()
    outputs = y_pred.numpy()
    levs = []
    for i in range(len(inputs)):
        # Convertit les séquences de reconstruction en noms
        input = "".join([idx2char[j] for j in inputs[i]])
        output = "".join([idx2char[j] for j in tf.argmax(outputs[i], axis=1).numpy()])
        levs.append(1+(lev(input, output)/len(input))**2)

    # Multiplie la perte selon la distance de Levenshtein
    levs = tf.reshape(tf.constant(levs, dtype=tf.float32), shape=(-1, 1))
    loss = tf.multiply(loss, levs)
    return loss


# Callback pour afficher l'avancement des noms
class ProgressCallback(keras.callbacks.Callback):
    def __init__(self, test_name):
        super().__init__()
        self.test_name = test_name[0]# + "^"
        name_as_int = [char2idx[c] for c in test_name]
        name_as_int = keras.preprocessing.sequence.pad_sequences([name_as_int], maxlen=seqLength, padding="post")
        self.name_as_int = tf.convert_to_tensor(name_as_int)
    
    def on_train_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            reconstructed_name = self.model(self.name_as_int)
            reconstructed_name = tf.squeeze(reconstructed_name, axis=0)

            # Convertir les séquences de reconstruction en noms
            decoded_name = [idx2char[i] for i in tf.argmax(reconstructed_name, axis=1).numpy()]
            decoded_name = "".join(decoded_name).split("^")[0]

            print()
            sys.stdout.write("\033[K")
            print("                         Reconstruction de James:", decoded_name)
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[F")



# EarlyStopping custom pour ne pas arrêter l'entraînement trop tôt
# (Le modèle a une longue periode de stagnation avant de commencer à s'améliorer)
class EarlyStoppingWithCheck(keras.callbacks.EarlyStopping):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs={}):
        # Execute the original on_epoch_end method
        super().on_epoch_end(epoch, logs)
        
        if self.model.stop_training:
            # Maintenant on teste manuellement quelques noms, juste pour s'assurer que le modèle ne retourne pas n'importe quoi
            testNames = ["marie helene", "andre joseph", "jean paul", "constantin", "james"]

            # Choisi un nom au hasard
            i = np.random.randint(0, len(testNames)-1)
                
            name_as_int = [char2idx[c] for c in testNames[i]]
            name_as_int = keras.preprocessing.sequence.pad_sequences([name_as_int], maxlen=seqLength, padding="post")
            name_as_int = tf.convert_to_tensor(name_as_int)

            reconstructed_name = self.model(name_as_int)
            reconstructed_name = tf.squeeze(reconstructed_name, axis=0)
            decoded_name = [idx2char[i] for i in tf.argmax(reconstructed_name, axis=1).numpy()]
            decoded_name = "".join(decoded_name).split("^")[0]

            print()
            print("Validation avec "+testNames[i]+" : ", decoded_name)

            if testNames[i] == decoded_name.strip().lower():
                print("Succès !")
            else:
                print("Échec !")
                #self.model.stop_training = False

# L'autotrainer
progressCallback = ProgressCallback(test_name="james")
iteration = 0
 
def buildAutoEncoder(latentDim=32, latentDimInterne=64, optimizerChoice="adam", learningRate=0, decaySteps=0, decayRate=0, dropoutRate=0.25, epochs=0, batch_size=0, validation_split=0):
    global iteration
    iteration += 1

    # Annonce le début de l'entraînement  
    print()
    print("============== Nouvelle itération ==============")
    print("                 Iteration: ", iteration, "/", nbIterations)
    print("latentDim:", latentDim)
    print("latentDimInterne:", latentDimInterne)
    print("optimizerChoice:", optimizerChoice)
    print("learningRate:", learningRate)
    print("decaySteps:", decaySteps)
    print("decayRate:", decayRate)
    print("dropoutRate:", dropoutRate)
    print("nbEpochs:", epochs)
    print("batchSize:", batch_size)
    print("validationSplit:", validation_split)
    print()
    print()

    # Encodeur
    encodeur = keras.Sequential([
        keras.layers.InputLayer(input_shape=[seqLength]),
        keras.layers.Embedding(vocabSize, latentDim),
        keras.layers.LSTM(latentDimInterne),
        keras.layers.Dropout(dropoutRate),
        keras.layers.Dense(latentDimInterne, activation="tanh"),
        keras.layers.Dense(latentDim, activation="tanh"),
    ], name="encodeur")

    
    # Décodeur
    decodeur = keras.Sequential([
        keras.layers.InputLayer(input_shape=[latentDim]),
        keras.layers.Dense(latentDimInterne, activation="tanh"),
        keras.layers.Dropout(dropoutRate),
        keras.layers.RepeatVector(seqLength),
        keras.layers.LSTM(latentDimInterne, return_sequences=True),
        keras.layers.Dense(vocabSize, activation="softmax")
    ])

    # Learning rate schedule
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learningRate,
        decay_steps=decaySteps,
        decay_rate=decayRate)

    # Compile le modèle
    ae = AutoEncoder(encodeur, decodeur)
    if optimizerChoice == 'Nadam':
        ae.compile(optimizer=keras.optimizers.Nadam(learning_rate=lr_schedule), loss=loss_fn)
    else:
        ae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss=loss_fn)

    return ae
 
##### Modèle AutoEncoder
if retrain:     
    # Early stopping    
    earlyStopping = EarlyStoppingWithCheck(
        patience=10, 
        restore_best_weights=True, 
        monitor="val_loss", 
        start_from_epoch=20
    )

    # Wrapper pour RandomizedSearchCV
    aeWrapper = KerasRegressor(build_fn=buildAutoEncoder)
    random_search = RandomizedSearchCV(
        estimator=aeWrapper,
        param_distributions=param_dist,
        verbose=2,
        n_iter=3,#nbIterations,
        refit=True,
        n_jobs=1
    )
    random_search.fit(nomsDataset, nomsDataset, callbacks=[progressCallback, earlyStopping])

    # Save the results for analysis
    results = pd.DataFrame(random_search.cv_results_)
    results.to_csv("all_results.csv", index=False)

    # Get the best hyperparameters
    best_params = random_search.best_params_
    print("Meilleurs params : ", best_params)

    # Sauvegarde l'encodeur et le décodeur
    best_model = random_search.best_estimator_
    print("Meilleur modele : ", best_model)
    best_model.save_weights("best_model.h5")

    ae = AutoEncoder(best_model.encodeur, best_model.decodeur)

else:
    # Encodeur
    encodeur = keras.Sequential([
        keras.layers.InputLayer(input_shape=[seqLength]),
        keras.layers.Embedding(vocabSize, param_dist['latentDim']),
        keras.layers.LSTM(param_dist['latentDimInterne']),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(param_dist['latentDimInterne'], activation="tanh"),
        keras.layers.Dense(param_dist['latentDim'], activation="tanh"),
    ], name="encodeur")
    
    # Décodeur
    decodeur = keras.Sequential([
        keras.layers.InputLayer(input_shape=[param_dist['latentDim']]),
        keras.layers.Dense(param_dist['latentDimInterne'], activation="tanh"),
        keras.layers.Dropout(0.25),
        keras.layers.RepeatVector(seqLength),
        keras.layers.LSTM(param_dist['latentDimInterne'], return_sequences=True),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(vocabSize, activation="softmax")
    ])
    ae = AutoEncoder(encodeur, decodeur)

 
################################# Test sur des noms ###################################
testNames = [
    "James", "Helene", "Eugenie", "Haxan", "Keven", "Marie Anne", "Joseph Arthur",
    "Jomes", "Helena", "Eugene", "Haxen", "Kevin", "Marieanne", "Jos Arture",
    "Morie", "Marie", "Morin", "Maria", "Maurice", "Bob", "Jean Benoit"
]
namesEncoded = []

# Encode et décode les noms de test
for name in testNames:
    name = name.lower()#+ "^"
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
    
# Pour chaque combinaisons de noms, calcule la distance entre les deux noms et affiche le résultat dans une matrice
print("Distances entre les noms:")
for i in range(len(testNames)):
    # Affiche les noms en haut de la matrice
    if i == 0:
        for name in testNames:
            print(name, end=";")
        print()
    for j in range(len(testNames)):
        if j == 0:
            print(testNames[i], end=";")
        dist = tf.norm(namesEncoded[i] - namesEncoded[j])
        print(dist.numpy(), end=";")
    print()

# Inscrit les résultats de la matrice dans un fichier CSV
with open("distances.csv", "w") as f:
    f.write("Nom1,Nom2,Distance\n")
    for i in range(len(testNames)):
        for j in range(len(testNames)):
            dist = tf.norm(namesEncoded[i] - namesEncoded[j])
            f.write(testNames[i]+","+testNames[j]+","+str(dist.numpy())+"\n")