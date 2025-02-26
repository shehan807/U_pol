#!/bin/bash

./gdma < psi4_dma_datafile > temp.dma
./parse_psiDMA_new.py temp.dma > temp_format.dma
./mpfit temp_format.dma > charges
