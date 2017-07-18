#! /bin/bash
# Generación de memoria TFG

latex -shell-escape memoria.tex 
bibtex memoria.aux 
latex -shell-escape memoria.tex 
latex -shell-escape memoria.tex 
dvips memoria.dvi 
ps2pdf memoria.ps 
