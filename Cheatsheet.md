# Cheatsheet
## Desicion Tree
    - Grayscale wird verwendet, um die Anzahl der Dimensionen zu reduzieren
    - shape wird verwendet um die Bilder auf die gleichen Dimensionen zu skalieren -> Verringerung der Features
    - flatten() verwandelt Bilder in shape zu einem langen Array/Vektor
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
-	ML Verfahren, dass sich an Biologischer Struktur orientiert
-	Neuron: 
  -	Atomare, einfache Einheit
	- Mehrere Inputs, die gewichtet miteinander Addiert werden (Aktivierungsfunktion) 
	- Output wird so errechnet und an weitere Neuronen weitergegeben
	- In schichten organisiert (Eingangs + Ausgangsschicht + Hidden Layers) 

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



