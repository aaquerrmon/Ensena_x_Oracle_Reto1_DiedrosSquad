# Libreria con funciones para no reutilizar código
import os
import random 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image

nClases = len(os.listdir(os.path.join(os.getcwd(), "train"))) # Number of classes = 8
labels = list(range(nClases))
labels_text = { 0 : "Hamburguesa", 1 : "Pollo Frito", 2 : "Donuts", 3 : "Patatas Fritas", 4 : "Hot Dog", 5 : "Pizza", 6 : "Sandwich-Bocata", 7 : "Patata Asada"}

        
def show_images(cantidad_imagenes, show_cantidad_imagenes = False):
    dirs = []
    files = []
    next = []
    for label in labels:
        dirs.append(os.path.join(os.getcwd(), 'train', str(label)))
        print(f'total training {label} images:', len(os.listdir(dirs[label]))) if show_cantidad_imagenes else None
        files.append(os.listdir(dirs[label]))
        pic_index = random.randint(0,len(os.listdir(dirs[label])))
        next.append(os.path.join(dirs[label], fname) for fname in files[label][pic_index-cantidad_imagenes:pic_index])
    
    fig = plt.figure(figsize=(30, 30))
    for label in labels: 
        for i, img_path in enumerate(next[label]):
            fig.add_subplot(nClases, cantidad_imagenes, 1+label*cantidad_imagenes+i) 
            img = mpimg.imread(img_path)
            plt.imshow(img)
            plt.axis('Off')
            plt.title(labels_text[label]) #,fontsize = 15)
    plt.show()
    
def generate_dataset(batch_size, target_size):
    TRAINING_DIR = os.path.join(os.getcwd(),"train")
    training_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    batch_size = batch_size

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(target_size,target_size),
        class_mode='categorical',
        batch_size=batch_size,
        subset="training"
    )

    val_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(target_size,target_size),
        class_mode='categorical',
        batch_size=batch_size,
        subset="validation"
    )
    return train_generator, val_generator

def callbacks(nombre_carpeta):
    # CALLBACKS PARA PARAR EL MODELO 
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('f1_score') > 0.95) & (logs.get('val_f1_score') < 0.9):
                print("\nEl F1_score de train es 0.95 y el de test es menor a 0.9. Se esta sobre-entrenando. Cancelamos el entrenamiento!")
                self.model.stop_training = True

    # Instantiate class
    callback = myCallback()

    callbackNoImprove = tf.keras.callbacks.EarlyStopping(monitor='f1_score', min_delta = 0.0005, patience=20, mode = "max", verbose = 1)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(nombre_carpeta,
                                                    monitor = "val_f1_score",
                                                    verbose = 1,
                                                    mode = 'max',
                                                    save_best_only=True)
    return [callback, callbackNoImprove, checkpoint]
    
def train_model(model, epochs, target_size, batch_size, nombre_carpeta):
    
    train_generator, test_generator = generate_dataset(batch_size, target_size)
    cb = callbacks(nombre_carpeta)
    # Set the training parameters
    model.compile(loss = 'categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=[tfa.metrics.F1Score(num_classes=nClases,
                                                                                                    average='macro', # Se puede estableces como None, micro, macro y weighted
                                                                                                    threshold=0.5)]) 
    # Train the model
    history = model.fit(
        train_generator, 
        validation_data = test_generator,
        epochs = epochs, 
        callbacks = cb
    )
    return model, history
 
def generate_predictionJSON(model):
    test = pd.read_csv("test.csv")
    # cargar las imágenes de prueba
    imagenes = []
    for path_img in test["path_img"]:
        img = image.load_img(path_img, target_size=(256, 256))
        x = image.img_to_array(img)
        imagenes.append(x)

    # preprocesar las imágenes
    imagenes = np.array(imagenes)
    imagenes = imagenes.astype('float32') / 255.0

    # hacer predicciones
    predicciones = model.predict(imagenes)

    # convertir las predicciones en una lista de clases
    clases = np.argmax(predicciones, axis=1)

    # crear un diccionario de índices de prueba y sus clases predichas
    diccionario = {}
    for i in range(len(clases)):
        diccionario[int(test["idx_test"][i])] = int(clases[i])

    # guardar el diccionario como un archivo JSON
    json_entrega = {
        "target": diccionario
    }

    json_object = json.dumps(json_entrega)

    with open("entrega.json", "w") as outfile:
        outfile.write(json_object)
