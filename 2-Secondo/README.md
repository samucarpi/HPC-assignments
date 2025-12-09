# Istruzioni

## Compilazione ed esecuzione base
Per compilare, eseguire e visualizzare il tempo di esecuzione, eseguire il seguente comando, sostituendo i segnaposto `{OTTIMIZZAZIONE}` e `{DATASET}` con le opzioni desiderate:
```bash
make clean; make FLAGS="-DPOLYBENCH_TIME -D{OTTIMIZZAZIONE} -D{DATASET}"; ./cholesky.exe
```
Tipi di ottimizzazione (`{OTTIMIZZAZIONE}`)
- SEQUENTIAL
- OPTIMIZED_v1
- OPTIMZED_v2

Tipi di dataset (`{DATASET}`)
- MINI_DATASET
- SMALL_DATASET
- STANDARD_DATASET
- LARGE_DATASET
- EXTRALARGE_DATASET

## Raccolta delle statistiche sui tempi di esecuzione
Per compilare, eseguire e raccogliere delle statistiche sui tempi è necessario eseguire il seguente script:
```bash
./benchmark.sh
```
Lo script richiederà in input il tipo di dataset su cui si vogliono eseguire le ottimizzazioni. Una volta inserito, avverranno 5 esecuzioni per ogni ottimizzazione dove, per ognuna di esse verrà mostrato il tempo migliore, peggiore e una media.
Le varie statistiche saranno poi raggruppate all'interno del file `benchmark.log`.

## Testing

Per testare che il programma svolga i calcoli correttamente, è necessario inserire una matrice nel file testuale `custom_matrix.txt` ed eseguire il seguente comando, sostituendo i segnaposto:
```bash
make clean; make FLAGS="-D{OTTIMIZZAZIONE} -DINIT_DEBUG -DPOLYBENCH_DUMP_ARRAYS -DN={DIMENSIONE_MATRICE}"; ./cholesky.exe
```

## Autori

 - Sebastiano Benatti (231589@studenti.unimore.it)
 - Samuele Carpi (301925@studenti.unimore.it)
 - Mattia Pasquali (386872@studenti.unimore.it)