# Cheatsheet
## Desicion Tree
    - Grayscale wird verwendet, um die Anzahl der Dimensionen zu reduzieren
    - shape wird verwendet um die Bilder auf die gleichen Dimensionen zu skalieren -> Verringerung der Features
    - flatten() verwandelt Features aus einer Matrix zu einem langen eindimensionalen Array/Vektor
    - tree.DecisionTreeClassifier() erstellt einen initialen Desicion Tree
    - mit fit() wird das model auf die Trainingsdaten angepasst
    - model.predict() gibt für Testdaten die erwartete Klasse aus

### DecisionTreeClassifier
- baut sich aus unterschiedlichen Merkmalen zusammen
- es gibt ein Wurzelelement, wo sich Blätter von abspalten
- tree.DecisionTreeClassifier() erstellt einen initialen Desicion Tree
  - random_state kontrolliert den Zufall beim Erstellen des Models
  - criterion beeinflusst das Kriterium, welches den Split bewertet (gini oder Entropy)
  - max_depth: maximale Tiefe des Trees
  - max_features: Anzahl betrachenden features
### Gini-Impurity
- Wahrscheinlichkeit, dass zwei unterschiedliche Ergebnisse erzielt werden können -> Unreinheit
### Entropy
- beschreibt wie gut/schlecht die Daten in einem Datensatz sortiert sind
- hohe Entropy: starke Vermischung

### RandomForestClassifier
- ist eine Vielzahl an Desicion Trees 
  - n_estimators gibt an wie viele Trees in dem Forest enthalten sind

## Crossvalidation
- wird dazu verwendet, um sicherzustellen, dass es keinen ungünstigen Split des Datensatzes gibt
- typischerweise wird eine k-Fold Crossvalidierung verwendet
  - k gibt an in wie viele Teile der Datensatz für die Crossvalidierung geteilt wird
  - mögliche Arten um Crossvalidierungen durchzuführen:
    - normal (Teile am Stück)
    - stratified (verteilt)
    - shuffled (random)

## Seaborn
- ist eine auf matplotlib basierende Visualisierungsbibliothek von Python
- wird verwendet um die Heatmap darzustellen
  - square=True &rarr; Felder der Heatmap sind Quadrate
  - annot=True &rarr; Werte in Heatmap
  - fmt='d' &rarr; Formatierung von Werten in der Heatmap zu Integer
  - cmap="YlGnBu" &rarr; Farbschema
  - linewidths=.5 &rarr; Abstand der Felder in der Heatmap
  - cbar_kws &rarr; Positionierung der Legende

## NumPy
- kann kompilierte mathematische und numerische Funktionen und Funktionalitäten in der größtmöglichen Ausführungsgeschwindigkeit durchführen
  - std &rarr; Standardabweichung
  - mean &rarr; Durchschnitt von den Werten in einem Array
  
## Confusion Matrix
- stellt die Zuordnung in einer Matrix dar
- Type 1 Error: False Positive
- Type 2 Error: False Negative
- Diagonale der Matrix gibt richtige Zuordnungen an
- wird benötigt für Kennzahlen wie Recall / Precision / Accuracy

## Recall
- wie viel % der richtigen "true" wurden als solche richtig erkannt?
- = $\frac{TP}{TP+FN}$
## Precision
- wie viel % der predicted "true" sind wirklich richtig "true"?
- = $\frac{TP}{TP+FP}$
## Accuracy
- wie viel % der kompletten Daten wurden richtig zugeordnet?
- = $\frac{TP+TN}{TP+TN+FP+FN}$

## F1-Score
- gut um verschiedene Models miteinander zu vergleichen
- misst recall und presicion zugleich
- = $\frac{2*Recall*Precision}{Recall+Presicion}$

## Tensorflow: 
-	Open-Source-Framework von Google, hauptsächlich Anwendung im ML & AI Bereich 
- Insbesondere Spracherkennung & Bildverarbeitung
-	Platformunabhängig, gut Skalierbar


## Keras 
-	Open-Source, Python, Deep-Learning Bibliothek
-	Einheitliche Schnittstelle zu verschiedenen Backends (Tensorflow, MS Cognitive Toolkit) => jetzt nurnoch Tensorflow
-	Soll Einstieg in diese Bibliotheken erleichetern
-	High-Level-API zur Implementierung von Neuronalen Netzwerken für Deep Learning
-	Nicht Deep-Learning Framework => interface für DeepLearning Frameworks wie Tensorflow

## Künstliche Neuronale Netze: 
-	ML Verfahren, dass sich an Biologischer Struktur, dem Neuron orientiert
-	Neuron: 
  -	Atomare, einfache Einheit
	- Mehrere Inputs, die gewichtet miteinander Addiert werden (Aktivierungsfunktion) 
	- Output wird so errechnet und an weitere Neuronen weitergegeben
	- In schichten organisiert (Eingangs + Ausgangsschicht + Hidden Layers) 
  
### Optimizer
- Every time a neural network finishes passing a batch through the network and generating prediction results, it must decide how to use the difference between the results it got and the values it knows to be true.
- It adjusts the weights on the nodes so that the network steps towards a solution.
- The algorithm that determines that step is known as the optimization algorithm.

### Convolutional Neural Network (CNN)
- erhält den Namen durch die Art der Hidden Layers im Netz (convolutional layers, pooling layers, fully connected layers, and normalization layers)
- Convolutional und Pooling Funktionen sind als Aktivierungsfunktionen verwendet
  - s. Conv2D + MaxPooling2D

### Recurrent Neural Network (RNN)
- kann gut Sequenzen verarbeiten
  - Schrift, Text, Sprache, ...
- kann im Gegensatz zu Feed-Forward Netzen Informationen an vorherige Schichten / die gleiche Schicht weitergeben

### ImageDataGenerator
- Bearbeiten von eingelesenen Bildern für die Verarbeitung im Modell (zB durch vgg16 Preprocessing function)
- Auch möglich: Image Augmentation direkt im Arbeitsspeicher -> Vervielfältigen der Bilder mit kleinen Abwandlungen
    - z.B. Rotieren, Rescalen, zoomen, spiegeln, farbanpassungen...

### DirectoryIterator
- Liest Bilder aus einem Verzeichnis ein
- Nimmt ImageDataGenerator entgegen
- Target_size für Reshape
- Color_mode als RGB, RGBA, Greyscale => Je nach dem Input auch (224, 224, 1/3/4)

### Keras Sequential Model
- Modell für einzelne, einfache Schichten => Jeweils einen Input/Output Tensor
- Input Shape muss an erstes Layer gegeben werden
- Nicht geeignet, wenn: 
    - Mehrere Inputs/Outputs pro Model/Layer
    - Layer-Sharing geplant ist
    - Man eine nicht-lineare Topology will (residual connection, multi-branch Model)
- Verhält sich wie Liste von Layer


#### Model.summary()
- gibt Zusammenfassung über Modell aus 
- Übersicht der Layer & Parameter
#### Model.compile()
- Konfiguriert den Lernprozess
- Nimmt Optimizer, Loss Function & Metrics entgegen
#### Model.fit()
- Trainiert das Modell
- Input Data & Labels als Numpy_array
- Epochs (anzahl Durchläufe), verbose (Logging), Callbacks (Callback Funktion), Validation Data (zum Validieren)

#### Metrics (Accuracy, Precision, Recall)
- Bei Model.compile mitgegeben
- Für Plotten eingesetzt

#### Early Stopping (Callback)
- Stoppt Training wenn überwachte Metric sich nicht mehr verbessert 
- min_delta: Minimum Wert, der noch als improvement zählt => Wenn wert unterschritten wird wird angehalten
- patience: Epochen, die nach keiner Verbesserung trotzdem noch durchlaufen

#### ModelCheckpoint (Callback)
-  Speichert Modell oder Weights in Angegebener Frequenz 

### Weitere Begriffe: 
   - Epoche -> Ein durchlauf aller Testdaten
   - Batch -> kleinere Einheit an Testdaten, die gleichzeitig betrachtet wird
   - verbose -> Einstellungen für das Logging
   - Layer (Conv2D, MaxPool2D, Flatten, Dense)

#### Conv2D 
  - Standard 2D-Convulistion Layer
  - erstellt convolution Kernel der über das Bild gelegt wird und outputs (tensor) erstellt
  - Input Image Matrix wird mit Kernal Matrix zu Output Matrix verrechnet
  - Der Kernel wendet filter auf die verschiedenen Bildbereiche an => Anzahld der filter in FUnktionsaufruf definiert  (potenzen von 2 als Anzahl, steigend je tiefer das layer)

### MaxPooling2D
  - Reduziert dimensionen des Outputs 
  - Keine Weights / Lernfortschritt 
  - Legt matrix mit angegebener PoolSize über filter und reduziert sie auf größte Zahl

### Flatten
 - Convertiert Matrix zu Array => hängt werte hintereinander

### Dense
 - Jede Input-Node ist mit jeder Output-Node verbunden => Dense/Dichtes Layer
 - Zur Transformation des Outputs
