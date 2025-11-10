#!/bin/bash

NUM_RUNS=5
OPTIMIZATION_FLAGS=("SEQUENTIAL" "REDUCTION" "ACCELERATOR" "TASKS")
DATASETS=("MINI" "SMALL" "STANDARD" "LARGE" "EXTRALARGE")
EXECUTABLE="./cholesky_acc"

RESULTS=()
SUMMARY_FILE="benchmark.log"

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

> $SUMMARY_FILE

#imposta il dataset
if [ "$DATASET_DEFINE_NAME" = "STANDARD" ]; then
  DATASET_DEFINE_OPTION=""
else
  DATASET_DEFINE_OPTION="-D${DATASET_DEFINE_NAME}_DATASET"
fi

# per ogni ottimizzazione 
for opt_flag in "${OPTIMIZATION_FLAGS[@]}"
do
  if [ "$opt_flag" == "SEQUENTIAL" ]; then
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

  RESULTS=()

  # esecuzioni
  for (( i=1; i<=$NUM_RUNS; i++ ))
  do
    time_taken=$( $EXECUTABLE 2>&1 )
    RESULTS+=("$time_taken")
    echo "Esecuzione $i/$NUM_RUNS: $time_taken s"
  done
  echo ""
  # statistiche
  echo "Statistiche"
  echo -e "\n--- $opt_flag ---" >> $SUMMARY_FILE
  awk '
    BEGIN {
      min = 9999999;
      max = 0;
      sum = 0;
    }
    {
      if ($1 < min) { min = $1 }
      if ($1 > max) { max = $1 }
      sum += $1
    }
    END {
        avg = sum / NR;
        print "---------------------------------";
        print "Esecuzioni Totali: " NR;
        print "Tempo Minimo (Best): " min " s";
        print "Tempo Massimo (Worst): " max " s";
        print "Tempo Medio (Avg):   " avg " s";
        print "---------------------------------";
    }
  ' <(printf "%s\n" "${RESULTS[@]}") | tee -a $SUMMARY_FILE
done