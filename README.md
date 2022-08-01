

<snippet>
  <content>
  
# Heritability_estimation_NGG

Heritability_estimation_NGG is a script designed to estimate heritability gain in multidimentional GWAS analysis.
	  
This code has been designed by CARLUER Jean-Baptiste for research purpose : "Link to the paper"

## Dependancy

- pandas
- numpy
- matplotlib

## Usage

This algorithm is design to estimate heritability gain from 2D ktop SNP's. The main goal is to compare 1D, 1D_random, 1D+2D, 1D_rand+2D_rand and 1D+2D_rand using PCR. PLS method is also implemented.

To run your own study : 

    python main.py [args]
    
List of mains args : 	  
	  
	-1D : the number of SNP to use from the ktop 1D file
	-2D : the number of interaction to use from the ktop 2D file
  	-adjusted : a boolean to indicate if the R-square should be adjusted
  	-nc : the maximum number of component.
	-x: the path to the genotype file to select which 1D SNP and 2D interactions to keep
  	-phenotype : a list of phenotype to use (this must be the same name as the phenotype dirname)
  	-out : location to save results
	-y : the path to the fir of phenotype file (if phenotype is As75 and located in data/Y/As75, you must give only data/Y/)
  	-theta1D : the path to the nth-element 1D (if phenotype is As75 and nth_elem is data/nth_elem/As75/diag.ktop, you must give only data/nth_elem/*/diag)
  	-theta2D : the path to the nth-element 2D (if phenotype is As75 and nth_elem is data/nth_elem/As75/inter.ktop, you must give only data/nth_elem/*/inter)
	  
## Extention informations
	  
We use .ktop to point at SNP index/interaction index in a upper-matrix and .kval to point to their values.
	
