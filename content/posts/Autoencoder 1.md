---
date: 2026-03-31T19:30:00
draft: "false"
title: Autoencoder
---

## Step 1: Suddivisione del dataset

La suddivisione del dataset è stata effettuata utilizzando la funzione `train_test_split` del modulo `sklearn.model_selection`, dopo aver caricato i dati con la libreria *pandas*. Il primo passo fondamentale è stato separare i record delle aziende classificate come "sane" (valore 0 nella colonna `class`) da quelle in fallimento (valore 1). 

Il motivo risiede nella natura stessa dell'**anomaly detection** tramite autoencoder: il modello deve essere addestrato esclusivamente su dati normali (le aziende sane) per imparare a ricostruire accuratamente la "normalità". In questo modo, in fase di test, il modello restituirà un errore di ricostruzione sensibilmente più alto quando incontrerà esempi di aziende prossime alla bancarotta, che verranno quindi identificate come anomalie.

Il dataset delle aziende sane è stato suddiviso destinando il 70% al *train set* e il restante 30% ripartito equamente tra *validation set* e *test set*. Poiché questi ultimi due devono contenere entrambe le categorie di aziende per una valutazione corretta, sono stati successivamente integrati con i dati delle aziende fallite e rimescolati. Per la valutazione finale, è stata preservata la colonna dei target per ogni set.

## Step 2: Pulizia dei dati

Per lavorare sul dataset *Polish Companies Bankruptcy* è stato necessario rimuovere i valori nulli dalle feature e procedere con una normalizzazione, fondamentale data la presenza di valori molto distanti dalla media.

Per la gestione dei valori mancanti è stato utilizzato **SimpleImputer**, che di default sostituisce i dati nulli con il valore medio della colonna. Questa strategia è efficace per colonne numeriche; in caso di variabili categoriche, si potrebbero adottare strategie diverse come il *most frequent* o il *constant*.

Per la normalizzazione è stato adottato lo **StandardScaler**, che esegue la trasformazione utilizzando lo *standard score*:

$$z= \frac{x-\mu}{\sigma}$$

Dove $\mu$ è il valore medio della colonna e $\sigma$ la deviazione standard (che indica quanto i dati siano dispersi rispetto alla media).

In pratica, per ogni colonna viene calcolato lo *z-score* per ogni singolo valore. Il risultato è che ogni feature viene centrata intorno allo zero, ma **non limitata** in un intervallo predefinito (a differenza di quanto accadrebbe con un **MinMaxScaler**).

È fondamentale che i parametri di **SimpleImputer** e **StandardScaler** siano calcolati esclusivamente sul set di *train* (*fitting*). La trasformazione viene poi applicata a tutti i set, ma utilizzando i parametri derivati dal training per evitare fenomeni di **data leakage**.

## Step 3: Definizione del modello

*(Inserire qui i dettagli sull'architettura del modello)*

## Step 4: Addestramento

Data la dimensione del dataset, è stata utilizzata una strategia di *data loading* per suddividere gli esempi in *batch*, ottimizzando l'uso della memoria. Durante l'addestramento (esteso per 100 epoche), i batch vengono elaborati dal modello: le feature vengono compresse nello spazio latente, decodificate per la ricostruzione e infine confrontate con i valori di ingresso per calcolare la *loss*. 

Come ottimizzatore è stato scelto **AdamW**, mentre come funzione di costo il **Mean Squared Error (MSE)**. Il calcolo della perdita avvia poi la *backpropagation* per l'aggiornamento dei pesi.

Per monitorare il processo, viene calcolato il valore medio della perdita per ogni epoca. Nello specifico, `features.size(0)` restituisce il numero di campioni $N_i$ nel batch corrente.

```python
for features, labels in train_loader:
        features = features.to(device)
        optimizer.zero_grad()
        reconstructed = autoencoder(features)
        loss = criterion_train(reconstructed, features)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * features.size(0)

    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    training_losses.append(epoch_train_loss)
```

La variabile `running_train_loss` accumula l'errore totale dell'epoca. Poiché `loss.item()` restituisce la perdita media del batch ($\bar{\mathcal{L}}_i$), moltiplichiamo questo valore per la dimensione del batch ($N_i$) per ottenere la somma degli errori del batch:

$$\text{Errore}_{batch} = \bar{\mathcal{L}}_i \cdot N_i$$

L'errore cumulativo totale ($L_{total}$) è la somma degli errori di tutti i $B$ batch:

$$L_{total} = \sum_{i=1}^B \left( \bar{\mathcal{L}}_i \cdot N_i \right)$$

Al termine dell'epoca, la perdita media finale ($L_{epoch}$) si ottiene dividendo l'errore totale per il numero complessivo di campioni ($N_{total}$):

$$L_{epoch} = \frac{L_{total}}{N_{total}}$$

Questa logica può essere sintetizzata nella formula:

$$\text{running\_test\_loss} = \sum_{i=1}^B \left(\sum_{j=1}^{N_{i}} loss_{i,j}\right)$$

Nel ciclo di validazione, eseguito dopo ogni epoca, viene valutata la capacità del modello di ricostruire report sia di aziende sane che fallite. Terminata questa fase, è necessario determinare una **threshold** (soglia) per identificare le anomalie.

L'idea è calcolare gli *anomaly scores* sul set di validazione e analizzarne la distribuzione. Qui l'anomaly score è definito come la **loss media per ciascun record**. Per ottenerlo, è stato usato un criterio di valutazione con `reduction='none'`, ottenendo una matrice con l'errore associato a ogni feature per ogni elemento del batch. Calcolando la media lungo le righe, si ricava il vettore degli anomaly scores individuali.

## Step 5: Classificazione

Il modello ricostruisce con precisione le aziende sane (errore basso), ma non quelle in fallimento. Se l'errore di ricostruzione supera una certa soglia, l'azienda viene classificata a rischio bancarotta. Il problema centrale è individuare la soglia ottimale. Le strategie testate sono tre:

1. Percentile
2. Gaussian Fit
3. Precision Recall Curve

### 3. Precision Recall Curve

Precision e Recall sono metriche fondamentali per valutare i modelli di classificazione:
- **Precision**: indica l'attendibilità del modello quando predice un'anomalia. Risponde alla domanda: *"Quando il modello segnala un'anomalia, quanto spesso ha ragione?"*
  $$P = \frac{TP}{TP+FP}$$
- **Recall**: misura la capacità di individuare i casi positivi reali. Risponde alla domanda: *"Quante delle aziende realmente in crisi sono state individuate?"*
  $$R = \frac{TP}{TP+FN}$$

!![Image Description](/images/Precision-Recall%201.png)

Nell'esempio sopra, vogliamo distinguere palline rosse da blu. Se il modello identifica 4 palline come rosse, ma solo 3 lo sono effettivamente su 5 totali:
- $P = 3/4 = 75\%$
- $R = 3/5 = 60\%$

È possibile variare la soglia che divide i positivi dai negativi per bilanciare precision e recall. Questo compito è affidato alla funzione `precision_recall_curve()` di `sklearn`. Confrontando gli score calcolati (le loss) con i valori di classe reali, cerchiamo l'errore di ricostruzione ideale per definire il fallimento.

Per esemplificare, analizziamo 4 record:

| Target ($y$) | Loss ($y^*$) |
| :--- | :--- |
| 0 | 0.0388 |
| 0 | 0.1552 |
| 0 | 0.2076 |
| 1 | 0.1097 |

Scegliendo come soglia il primo valore di loss ($0.0388$), tutti i record con loss $\ge 0.0388$ vengono classificati come anomalie ($\hat{y}=1$):

| Target ($y$) | Loss ($y^*$) | Classificazione ($\hat{y}$) |
| :--- | :--- | :--- |
| 0 | 0.0388 | 1 |
| 0 | 0.1552 | 1 |
| 0 | 0.2076 | 1 |
| 1 | 0.1097 | 1 |

Calcolando le metriche: $P = 1/4 = 25\%$ e $R = 1/1 = 100\%$. Iterando questo processo per ogni valore di loss, otteniamo la curva Precision-Recall.

!![Image Description](/images/precision-recall-curve.png)

Per trovare la soglia perfetta introduciamo l'**F1-score**, la media armonica tra precision e recall:

$$F1=2\frac{P\cdot R}{P+R}$$

La media armonica penalizza i valori estremi: l'F1-score sarà alto solo se sia precision che recall sono soddisfacenti. Individuato il valore massimo di F1-score nella curva, utilizziamo l'indice corrispondente per determinare la nostra threshold definitiva.

```python
optimal_idx = np.argmax(f1_scores)

optimal_threshold = thresholds[optimal_idx]
```

## Note

- **F1 Score**: Fondamentale per bilanciare il modello, specialmente con classi sbilanciate.
- **Scelta della metrica**: In alcuni contesti potremmo preferire una Recall più alta (per non perdere nessuna azienda in crisi) anche a costo di una Precision inferiore.
- **Efficienza**: Sebbene spiegati passo-passo, i calcoli avvengono tramite operazioni vettoriali tra matrici, sfruttando l'efficienza computazionale dei tensori.
