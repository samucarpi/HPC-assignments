#!/bin/bash

NUM_RUNS=5 # numero di esecuzioni per ogni ottimizzazione
OPTIMIZATION_FLAGS=("SEQUENTIAL" "REDUCTION" "ACCELERATOR" "TASKS") # ottimizzazioni disponibili
DATASETS=("MINI" "SMALL" "STANDARD" "LARGE" "EXTRALARGE") # dataset disponibili
EXECUTABLE="./cholesky_acc" # eseguibile

RESULTS=() # array per i risultati
SUMMARY_FILE="benchmark.log" # file di log
> $SUMMARY_FILE # pulizia del file di log

# richiesta input del dataset
echo "Inserisci la macro della dimensione del dataset (MINI, SMALL, STANDARD, LARGE, EXTRALARGE):"
while true; do
  read -p "> " DATASET_DEFINE_NAME
  DATASET_DEFINE_NAME="${DATASET_DEFINE_NAME^^}" # converte in uppercase
  DATASET_DEFINE_NAME="${DATASET_DEFINE_NAME// /}" # rimuove spazi

  check=false
  for v in "${DATASETS[@]}"; do
    if [ "$v" = "$DATASET_DEFINE_NAME" ]; then
      check=true
      break
    fi
  done
  if [ "$check" = true ]; then
    break
  else
    list=$(printf "%s, " "${DATASETS[@]}") # crea una stringa con le opzioni
    list=${list%, } # rimuove l'ultima virgola e spazio
    echo "Dataset non valido. Scegli uno tra: $list"
  fi
done

#imposta il dataset
echo -e "--- DATASET ---\n$DATASET_DEFINE_NAME" >> $SUMMARY_FILE # log del dataset scelto
if [ "$DATASET_DEFINE_NAME" = "STANDARD" ]; then
  DATASET_DEFINE_OPTION=""
else
  DATASET_DEFINE_OPTION="-D${DATASET_DEFINE_NAME}_DATASET"
fi

# per ogni ottimizzazione
for opt_flag in "${OPTIMIZATION_FLAGS[@]}"; do
  if [ "$opt_flag" = "SEQUENTIAL" ]; then
    echo ""
    echo "======================================================"
    echo "--- $opt_flag"
    echo "======================================================"
  else
    echo ""
    echo "======================================================"
    echo "--- Ottimizzazione: $opt_flag"
    echo "======================================================"
  fi

  # compilazione
  CFLAGS_BUILD="-fopenmp -DPOLYBENCH_TIME -D${opt_flag} ${DATASET_DEFINE_OPTION}"
  make clean > /dev/null 2>&1
  make CFLAGS="$CFLAGS_BUILD" $EXECUTABLE > /dev/null 2>&1
  if ! make CFLAGS="$CFLAGS_BUILD" $EXECUTABLE > build.log 2>&1; then
    echo "Compilazione fallita per $opt_flag"
    exit 1
  fi

  # esecuzioni
  RESULTS=()
  for ((i=1; i<=$NUM_RUNS; i++)) do
    time_taken=$( $EXECUTABLE 2>&1 )
    if [[ "$time_taken" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
      RESULTS+=("$time_taken")
      echo "Esecuzione $i/$NUM_RUNS: $time_taken s"
    else
      echo "Output non valido: $opt_flag esecuzione $i"
      exit 1
    fi
  done
  echo ""

  # statistiche
  echo "Statistiche"
  echo -e "\n--- $opt_flag ---" >> $SUMMARY_FILE
  awk '
    BEGIN {
      min = 1e99;
      max = 0;
      sum = 0;
    }
    {
      if ($1 < min) { min = $1 }
      if ($1 > max) { max = $1 }
      sum += $1
    }
    END {
      if (NR>0) {
        avg = sum / NR;
        print "---------------------------------";
        print "Esecuzioni Totali: " NR;
        print "Tempo Minimo (Best): " min " s";
        print "Tempo Massimo (Worst): " max " s";
        print "Tempo Medio (Avg):   " avg " s";
        print "---------------------------------";
      } else {
        print "Nessun dato valido raccolto.";
      }
    }
  ' <(printf "%s\n" "${RESULTS[@]}") | tee -a $SUMMARY_FILE
done