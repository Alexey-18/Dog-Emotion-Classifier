import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.preprocessing import image
import matplotlib.cm as cm
import random
import json
import cv2 
from datetime import datetime
import warnings
import math
from tensorflow.keras.models import Model

warnings.filterwarnings('ignore')

#Set random seeds for reproducibility
def set_random_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_random_seeds(42)

#Configuration
CONFIG = {
    'base_dir': "dog-emotion",
    'imege_size': (300, 300),
    'batch_size':16,
    'num_classes': 4,
    'epochs_initial': 25,
    'epochs_fineture': 25,
    'learning_rate_initial': 0.001,
    'leaning_rate_fineture': 1e-5,
    'validation_split': 0.2,
    'random-seed': 42,
    'valid_emotions': ['angry', 'sad', 'happy', 'relaxed']
}

print(" Dog Emotion Classifier")
print("-" *50)
print(f"Configuration: {json.dumps(CONFIG, indent=2)}")

#Data pipeline
def create_emotion_only_generators(base_dir, valid_emotions, config):
    existing_emotions = []
    for emotion in valid_emotions:
        emotion_path = os.path.join(base_dir, emotion)
        if os.path.isdir(emotion_path):
            existing_emotions.append(emotion)
            print(f" Found emotion folder: {emotion}")
        else:
            print(f" Missing emotion folder: {emotion}")

    #Training data generator with advanced augmentation
    train_datagen= ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=config['validation_split'],
        #Geometric augmentations
        rotation_range=30,
        wigth_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        shear_range=0.12,
        horizontal_flip=True,
        # Photometric augmentations
        brightness_range=[0.8, 1.2],
        channel_shift_range=0.1,
        fill_mode='nearest'
    )

    #Validation data genetator (no augmentation)
    val_datagen =ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=config['validation_split']
    )
    #Create generators - they should automatically ignore non-image folders
    train_gen = train_datagen.flow_from_directory(
        base_dir,
        target_size=config['image_size'],
        batch_size=config['batch_size'],
        class_mode='categorical',
        subset='training',
        shuffle=True,
        sed=config['random_seed'],
        classes=existing_emotions
    )

    val_gen = val_datagen.flow_from_directory(
        base_dir,
        target_size=config['image_size'],
        batch_size=config['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=config['random_seed'],
        classes=existing_emotions
    )

    print(f" Training generator created with classes: {train_gen.class_indices}")
    print(f" Validation generator created with classes: {val_gen.class_indices}")

    return train_gen, val_gen, existing_emotions

#Create data generators
train_gen, val_gen, existing_emotions = create_emotion_only_generators(CONFIG['base_dir'], CONFIG['valid_emotions'], CONFIG)

#Update config with actual number of classes found
CONFIG['num_classes'] = len(existing_emotions)
print(f"Updated number of classes: {CONFIG['num_classes']}")

#Model Architecture

def doggo_model(config):
    """Create an advanced model with nulti-scale feateres"""
    #Base model
    base_model = EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=(*config['image_size'], 3)
    )
    base_model.trainable = False

    #Multi-scale features from different scales
    inputs = base_model.input

    #Extract features from different scales
    low_level = base_model.get_layer('block2a_expand_activation').output #75x75
    mid_level = base_model.get_layer('bloack4a_expand_activation').output #19x19
    high_level = base_model.output #10x10

    #Process each scale
    low_features = layers.GlobalAveragePooling2D(name='low_gap')(low_level)
    mid_features = layers.GlobalAveeagePooling2D(name='mid_gap')(mid_level)
    high_features = layers.GlobalAveeagePooling2D(name='high_gap')(high_level)

    #Combine multi-scale features
    combined_features = layers.Concatenate(name='feature_concat')([
        low_features, mid_features, high_features
    ])

    #Custom Head
    x = layers.Dense(1024, activation='swish', name='dense_1014', kernel_regularizer=regularizers.l2(0.001))(combined_features)
    x = layers.BatchNormalization(name='bn_1024')(x)
    x = layers.Dropout(0.5, name='dropout_1')(x)
    x = layers.Dense(512, activation='swish', name='dense_512', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization(name='bn_512')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)

    #Output layer
    outputs = layers.Dense(config['num_classes'], activation='softmax', name='predictions')(x)
    model = models.Model(inputs=inputs, outputs=outputs, name='DogEmotionClassifier')
    return model

#Create model
model = doggo_model(CONFIG)
model.summary()

#Training Setup
def doggo_callbacks(model_name='best_dog_emotion_model.h5'):
    """Cheate comprehemsive callbacks for training"""

    callbacks_list = [
        #Early stopping
        callbacks.EarlyStopping(
            model='val_loss',
            patience=10,
            restore_vest_weights=True,
            verbose=1
        ),

        #Learning rate reduction
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),

        #Model checkpoint
        callbacks.ModelCheckpoint(
            filepath=model_name,
            monitor='val_accuracy',
            save_best_only=True,
            model='max',
            verbose=1
        ),

        #CSV logger
        callbacks.CSVLogger('training_log.csv'),

        #Learning rate scheduler
        callbacks.LearningRateScheduler(
            lambda epoch: CONFIG['learning_rate_initial'] * (0.95 ** epoch)
        )
    ]

    return callbacks_list

# Compile model
model.compile(
    optimizer=optimizers.AdamW(learning_rate=CONFIG['learning_rate_initial']),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Get callvacks
callbacks_list = doggo_callbacks()

#Initial Training (Feature Extration)
print(" Starting initial training (feature extraction)...")
initial_history = model.fit(
    train_gen,epochs=CONFIG['epochs_initial'],
    validation_data=val_gen,
    callbacks=callbacks_list,
    verbose=1
)

#FineTuning
def unfreeze_model(model):
    """Unfreeze model layers for fine-tuning"""
    #We unfreeze the top 100 layers (excludding BatchNormalization layers)
    for layer in model.layers[-100:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    #Recompile model with lower learning rate
    model/compile(
        optimizer=optimizers.AdamW(learning_rate=CONFIG['leaning_rate_fineture']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

#Unfreeze and recompile model 
model = unfreeze_model(model)
print(" Unreezing layers for fine-tuning...")

#Continue training
print(" Starting fine-tuning...")
fine_tune_history = model.fit(
    train_gen,
    initial_epoch=initial_history.epoch[-1] + 1,
    epochs=initial_history.epoch[-1] + 2 + CONFIG["epochs_fineture"],
    validation_data=val_gen,
    callbacks=callbacks_list,
    verbose=1
)

#Combine histories
histories = [initial_history,fine_tune_history]
fine_tune_start_epoch = initial_history.epoch[-1] + 1

#Plotting Loss and Accuracy
def plot_beautiful_training_charts(histories, fine_tune_start_epoch):
    """BEautiful charts: accuracy, loss, and enhanced summary with highest val accuracy marced."""

    #Combine histories 
    combined_history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'lr': []
    }

    for history in histories:
        for key in combined_history:
            if key in history.history:
                combined_history[key].extend(history.history[key])

    epochs = list(range(1, len(combined_history['loss']) + 1))
    total_epochs = len(epochs)

    #Find highest validation accuracy
    best_val_acc = max(combined_history['val_accuracy'])
    best_val_epoch = combined_history['val_accuracy'].index(best_val_acc) + 1
    best_vall_loss = combined_history['val_loss'][best_val_epoch - 1]

    #Get final metrics
    final_train_acc = combined_history['accuracy'][-1]
    final_train_loss = combined_history['loss'][-1]
    final_val_acc = combined_history['val_accuracy'][-1]
    final_val_loss = combined_history['val_loss'][-1]

    #Calculate improvement metrics
    accuracy_improvement = final_train_acc - combined_history['accuracy'][0]
    loss_improvement = combined_history['loss'][0] - final_train_loss

    #Set global style to match evaluation theme
    sns.set_style("whitegrid", {
        'axes.edgecolor': '0.4',
        'axes.labelcolor': '0.2',
        'axes.titleweight': 'bold',
        'grid.color': '0.92',
        'xtick.color': '0.4',
        'ytick.color': '0.4',
    })

    #Color palette
    PALETTE = ['#A0C4FF', "#BDB2FF", '#FFC6FF', '#FFADAD', '#CAFFBF', "#FDFFB5"]

    #Accuracy Chart
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, combined_history['accuracy'], '-', color=PALETTE[0], linewidth=2.5, alpha=0.9, label='Train Accuracy')
    plt.plot(epochs, combined_history['val_accuracy'], '-', color=PALETTE[1], linewidth=2.5, alpha=0.9, label='Val Accuracy')
    plt.scatter(best_val_epoch, best_val_acc, s=150, color='gold', edgecolor='black', zorder=5, linewidth=1.5, label=f'🏅 Best Val Acc ({best_val_acc:.2%}) at Epoch {best_val_epoch}')
    plt.axvline(fine_tune_start_epoch, color=PALETTE[4], linestyle='--', linewidth=2.5, alpha=0.8, label='Fine-tuning Start')
    plt.title('📈 Accuracy Over Epochs', fontsize=18, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(frameon=True, framealpha=0.9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    #Loss chart
    plt.figure(figsize=(10,6))
    plt.plot(epochs, combined_history['loss'], '--', color=PALETTE[0], linewidth=2.5, alpha=0.9, label='Train Loss')
    plt.plot(epochs, combined_history['val_loss'], '--', color=PALETTE[1], linewidth=2.5, alpha=0.9, label='Val Loss')
    plt.scatter(best_val_epoch, best_vall_loss, s=150, color='gold', edgecolor='black', zorder=5, linewidths=1.5,label=f'🏅 Val Loss at Best Acc ({best_vall_loss:.4f})')
    plt.axvline(fine_tune_start_epoch, color=PALETTE[4], linestyle='--', linewidth=2.5, alpha=0.8, label='Fine-uning Start')
    plt.title('📉 Loss Over Epochs', fontsize=18, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(frameon=True, framealpha=0.9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    #Summary
    bg_color = '#E6FCF5' #Light mint green
    border_color = PALETTE[4] #Darker mint green

    summary_text = f"""
    🚀 MODEL TRAINING SUMMARY
    
    ⭐ Peak Performance
    {'-'*45}
    🏁 Highest Validation Accuracy:  {best_val_acc:.4f}  (Epoch {best_val_epoch})
    🔍 Corresponding Validation Loss: {best_vall_loss:.4f}
    
    🏁 Best Validation Loss:          {min(combined_history['val_loss']):.4f}  
      (Epoch {combined_history['val_loss'].index(min(combined_history['val_loss'])) + 1})
    
    📊 Final Epoch Metrics (Epoch {total_epochs})
    {'-'*45}
    🧠 Training Accuracy:             {final_train_acc:.4f}
    📉 Training Loss:                 {final_train_loss:.4f}
    
    🎯 Validation Accuracy:           {final_val_acc:.4f}
    📊 Validation Loss:               {final_val_loss:.4f}
    
    🔄 Training Progress
    {'-'*45}
    📈 Accuracy Improvement:          +{accuracy_improvement:.4f}
    📉 Loss Reduction:                -{loss_improvement:.4f}
    
    ⚙️ Training Configuration
    {'-'*45}
    🔄 Total Epochs Trained:          {total_epochs}
    🛠️ Fine-tuning Started At:        Epoch {fine_tune_start_epoch}
    {' '}
    """

    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor('#F8FFFD') #Very pale mint background

    #Create centered text
    plt.figtext(0.5, 0.5, summary_text,fontsize=14,fontweight='bold',color='#2E4057',verticalalignment='center',horizontalaligment='center',bbox=dict(boxstyle="round,pad=1",facecolor=bg_color,edgecolor=border_color,linewidth=3,alpha=0.95))
    plt.suptitle('TRAINING PERFORMANCE SUMMARY',fontsize=22,fontweight='bold',color='#2E4A62',y=0.92)

    #Add decorative elements
    plt.annotate('★', (0.15, 0.85), fontsize=30, color='gold', alpha=0.7)
    plt.annotate('★', (0.85, 0.85), fontsize=30, color='gold', alpha=0.7)
    plt.annotate('★', (0.15, 0.15), fontsize=30, color='gold', alpha=0.7)
    plt.annotate('★', (0.85, 0.15), fontsize=30, color='gold', alpha=0.7)

    plt.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95]) #Make room for suplitle
    plt.show()

plot_beautiful_training_charts(histories, fine_tune_start_epoch)

#Recreate only the validation generator with shuffle=True
#So we can get Score Cam results of multiple emotion because our images are in individual folders
val_datagen = ImageDataGenerator(
    preprocess_function=preprocess_input,
    validation_split=CONFIG['validation_split']
)

val_gen = val_datagen.flow.from_directory(
    CONFIG['base_dir'],
    targer_size=CONFIG['image_size'],
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    subset='validation',
    shuffle=True, #shuffling validation data
    seed=CONFIG['random-seed'],
    classes=existing_emotions
)

def get_scorecam_layer(model):
    """Cose a valid conv layer for Score-CAM"""
    for layer_name in ['block4a_expand_activation', 'block2a_expand_activation', 'top_activation']:
        try:
            model.get_layer(layer_name)
            print(f" Usign Score-CAM layer: {layer_name}")
            return layer_name
        except:
            continue
    for layer in reversed(model.layers):
        if 'expand_activation' in layer.name:
            print(f" Fallback to: {layer.name}")
            return layer_name
    raise ValueError(" No valid layer found for Score-CAM")

def visualize_scorecam(model, val_gen, num_samples=12):
    """Score-CAM (resizing fully fized)"""
    total_subplots = num_samples * 2
    cols = 4
    rows = math.ceil(total_subplots / cols)
    plt.figure(figsize=(5 * cols, 5 * rows))
    
    layer_name = get_scorecam_layer(model)
    activation_model = Model(inputs=model.input, outputs=[
        model.get_layer(layer_name).output
    ])
    class_names = list(val_gen.class_indices.keys())
    val_gen.reset()
    x_batch, y_batch = next(val_gen)
    indices = random.sample(range(len(x_batch)), min(num_samples,len(x_batch)))

    for i, idx in enumerate(indices):
        try:
            img = x_batch[idx]
            x = np.expand_dims(img, axis=0)
            pred = model.predict(x, verbose=0)
            pred_class = np.argmax(pred[0])
            true_class = np.argmax(y_batch[idx])
            activation = activation_model.predict(x)[0]
            img_height, img_width = img.shape[0], img.shape[1]
            scorecam_map = np.zeros((img_height, img_width))
            norm_img = (img - np.min(img)) / (np.max(img) - np.min(img))
            norm_img = np.expand_dims(norm_img, axis=0)

            for ch in range(activation.shape[-1]):
                act = activation[..., ch]
                act = np.maximum(act, 0)
                act /= (act.max() + 1e-10)
                upsampled = cv2.resize(act, (img_width, img_height))
                mask = np.expand_dims(upsampled, axis=-1)
                masked_input = norm_img * mask
                score = model.predict(masked_input, verbose=0)[0][pred_class]
                scorecam_map += score * upsampled

            scorecam_map = np.maximum(scorecam_map, 0)
            scorecam_map /= np.max(scorecam_map)
            heatmap = cv2.applyColorMap(np.uint8(255 * scorecam_map), cv2.COLORMAP_JET)

            img_unit8 = np.unit8(255 * norm_img[0])
            superimposed = cv2.addWeighted(img_unit8, 0.6, heatmap, 0.4, 0)

            #Original
            plt.subplot(rows, cols, 2*i + 1)
            plt.imshow(img_unit8[..., ::-1])
            plt.title(f"Original\nTrue: {class_names[true_class]}")
            plt.axis('off')

            #Score-CAM
            plt.subplot(rows, cols, 2*i + 2)
            plt.imshow(superimposed[..., ::-1])
            plt.title(f"Score-CAM\nPred: {class_names[pred_class]}", color='green' if pred_class == true_class else 'red')
            plt.axis('off')

        except Exception as e:
            print(f" Error on index {idx}: {e}")
            continue

    plt.tight_layout()
    plt.show()
    print(" Score-CAM complete!")

#Run it
print(" Score-CAM starting...")
visualize_scorecam(model, val_gen)
