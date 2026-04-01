---
date: 2026-03-31T19:30:00
draft: "false"
title: Autoencoder
---

## Step 1: Suddivisione del dataset

La suddivisione del dataset è stata effettuata mediante la funzione `train_test_split` del modulo `sklearn.model_selection`, dopo aver caricato i dati con la libreria *pandas*. Il primo passo di questa fase è stato separare i record delle aziende classificate come "sane" (valore 0 nella colonna `class`) da quelle in fallimento (valore 1). Il motivo risiede nel fatto che, per effettuare *anomaly detection* con un autoencoder, è necessario addestrare il modello esclusivamente su dati normali; in questo modo, il modello imparerà a ricostruire accuratamente la "normalità", restituendo un errore di ricostruzione sensibilmente più alto quando, in fase di test, incontrerà esempi di aziende prossime alla bancarotta.

Il dataset delle aziende sane è stato quindi suddiviso: il 70% è stato destinato al *train set*, mentre il restante 30% è stato ripartito equamente (50% ciascuno) tra *validation set* e *test set*. Poiché questi ultimi due devono contenere entrambe le categorie di aziende per una valutazione corretta, sono stati successivamente integrati con i dati delle aziende fallite e rimescolati. Per la valutazione finale, è stata preservata la colonna dei target per ogni set.

## Step 2: Pulizia dei dati

Per lavorare sul dataset *Polish Companies Bankruptcy* è stato necessario un intervento di eliminazione dei valori nulli dalle colonne delle features e una successiva normalizzazione, necessaria a causa di valori che si discostavano eccessivamente dal valore medio.

Per la rimozione dei valori nulli è stato utilizzato **SimpleImputer**, che di default sostituisce i valori mancanti con il valore medio della colonna. È importante sottolineare che questa strategia è valida solo per colonne con valori numerici; in caso di valori categorici, si potrebbe adottare la strategia del *most frequent* o del *constant*.

Per la normalizzazione, su consiglio del docente, è stato adottato lo **StandardScaler**, che esegue una normalizzazione usando lo *standard score*:

$$z= \frac{x-\mu}{\sigma}$$

Dove $\mu$ è il valore medio della colonna e $\sigma$ la deviazione standard (ovvero di quanto, in media, i valori si allontanano dal valore medio: una deviazione standard bassa indica dati simili tra loro, mentre una alta indica dati molto diversi).

In pratica, per ogni colonna delle features vengono calcolate la deviazione standard e il valore medio. A seguire viene calcolato per ogni singolo valore lo *z-score*, che rappresenta il nuovo valore scalato. Il risultato finale è che ogni colonna viene centrata intorno allo zero, ma **non limitata** in un intervallo predefinito (risultato che si otterrebbe invece usando un **MinMaxScaler**).

È fondamentale notare come i parametri usati da **SimpleImputer** e **StandardScaler** debbano essere calcolati esclusivamente sul set di *train* (fase di *fitting*). La trasformazione vera e propria viene eseguita successivamente su tutti i set (*train*, *validation* e *test*), ma utilizzando esclusivamente i parametri derivati dal *train* per evitare fenomeni di *data leakage*.

## Step 3: Definizione del modello

## Step 4: Addestramento

Data l'entità del dataset, è stata utilizzata una strategia di *data loading* per suddividere gli esempi in *batch*, evitando così di sovraccaricare le risorse hardware. Durante la fase di addestramento (estesa per 100 epoche), i *batch* vengono caricati e le *features* elaborate dal modello: queste vengono scompattate, compresse, ricostruite e infine confrontate con i valori di ingresso per calcolare la *loss*. Come ottimizzatore è stato scelto **AdamW**, mentre come funzione di costo è stato adottato il **Mean Squared Error (MSE)**. A questo punto ha inizio la fase di *backpropagation* per l'aggiornamento dei pesi del modello.

Per monitorare l'andamento del processo, invece di visualizzare la *loss* per ogni singolo record, viene calcolato il valore medio della perdita per ogni epoca. Nello specifico, `features.size(0)` restituisce il numero di campioni $N_i$ presenti nel batch corrente.

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

La variabile `running_train_loss` accumula l'errore totale dell'intera epoca. Poiché `loss.item()` restituisce la perdita media del batch corrente ($\bar{\mathcal{L}}_i$), per ottenere la somma degli errori di tutti i campioni del batch si moltiplica tale valore per la dimensione del batch stesso ($N_i$):

$$\text{Errore}_{batch} = \bar{\mathcal{L}}_i \cdot N_i$$

L'errore cumulativo totale per l'intera epoca ($L_{total}$) è quindi la somma degli errori di tutti i $B$ batch:

$$L_{total} = \sum_{i=1}^B \left( \bar{\mathcal{L}}_i \cdot N_i \right)$$

Al termine dell'epoca, la perdita media finale ($L_{epoch}$) viene calcolata dividendo l'errore totale accumulato per il numero complessivo di campioni nel dataset di *training* ($N_{total}$):

$$L_{epoch} = \frac{L_{total}}{N_{total}}$$

Questo valore viene infine salvato per il successivo *plotting* dell'andamento dell'addestramento.

Le precedenti formule possono anche essere riscritte nel seguente modo:

$$\text{running\_test\_loss} = \sum_{i=1}^B \left(\sum_{j=1}^{N_{i}} loss_{i,j}\right)$$

dove:
- $i$ = indice del batch
- $j$ = elemento dentro il batch
- $B$ = numero di batch
- $N_i$ = numero di elementi nel batch $i$

Nel ciclo di validazione, eseguito al termine della fase di addestramento di ogni epoca, viene valutata la capacità del modello di ricostruire i report forniti, comprendenti sia aziende sane che aziende prossime alla bancarotta. Anche in questo caso, viene calcolata la *loss* media per ogni epoca seguendo la stessa logica di normalizzazione.

Una volta completate le fasi di addestramento e validazione, il passo successivo consiste nel determinare una *threshold* adeguata per valutare la capacità del modello di individuare anomalie.

L'idea di base è calcolare gli *anomaly scores* sul set di validazione, analizzarne la distribuzione (ad esempio tramite un grafico) e utilizzarli per definire la soglia decisionale. In questo caso, l'anomaly score è stato definito come la **loss media per ciascun record** del dataset.

Per ottenere questo risultato, è stato necessario introdurre un criterio di valutazione separato, impostando il parametro `reduction='none'`. In questo modo, invece di ottenere un unico valore medio per batch, si ricava una matrice che contiene, per ogni elemento del batch, l'errore associato a ciascuna feature.

Successivamente, si applica una riduzione calcolando la media lungo le righe della matrice, ottenendo così un vettore in cui ogni elemento rappresenta l'errore medio di ricostruzione per una singola transazione, ovvero il relativo anomaly score.

## Nota

Per quanto possa sembrare controintuitivo, i calcoli nei vari cicli non servono a prendere elemento per elemento, bensì, quello che avviene è un prodotto tra matrici che contengono o parti del dataset grazie al loader o tutto il dataset stesso

![[DF5E72E0-8610-452A-8434-02FE55D03AA6.jpg]]