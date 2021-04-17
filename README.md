# ECLARE
<<<<<<< HEAD
## [ECLARE: Extreme Classification with Label Graph Correlations](http://manikvarma.org/pubs/mittal21b.pdf)
```bib
@InProceedings{Mittal21b,
	author       = "Mittal, A. and Sachdeva, N. and Agrawal, S. and Agarwal, S. and Kar, P. and Varma, M.",
	title        = "ECLARE: Extreme classification with label graph correlations",
	booktitle    = "Proceedings of The ACM International World Wide Web Conference",
	month = "April",
	year = "2021",
	}
```

#### SETUP WORKSPACE
```bash
=======
ECLARE: Extreme Classification with Label Graph Correlations

#### SETUP WORKSPACE
```
>>>>>>> 49fc87b7eff8ac633cc3a851c34d7dfc3a6c7d6d
mkdir -p ${HOME}/scratch/XC/data 
mkdir -p ${HOME}/scratch/XC/programs
```

#### SETUP ECLARE
<<<<<<< HEAD
```bash
=======
```
>>>>>>> 49fc87b7eff8ac633cc3a851c34d7dfc3a6c7d6d
cd ${HOME}/scratch/XC/programs
git clone https://github.com/Extreme-classification/ECLARE.git
conda create -f ECLARE/eclare_env.yml
conda activate eclare
git clone https://github.com/kunaldahiya/pyxclib.git
cd pyxclib
python setup.py install
cd ../ECLARE
```

#### DOWNLOAD DATASET
<<<<<<< HEAD
```bash
=======
```
>>>>>>> 49fc87b7eff8ac633cc3a851c34d7dfc3a6c7d6d
cd ${HOME}/scratch/XC/data
gdown --id <dataset id>
unzip *.zip
```
<<<<<<< HEAD
| dataset                   | dataset id                        |
|---------------------------|-----------------------------------|
| LF-AmazonTitles-131K      | 1VlfcdJKJA99223fLEawRmrXhXpwjwJKn |
| LF-WikiSeeAlsoTitles-131K | 1edWtizAFBbUzxo9Z2wipGSEA9bfy5mdX |
| LF-AmazonTitles-1.3M      | 1Davc6BIfoTIAS3mP1mUY5EGcGr2zN2pO |

#### RUNNING ECLARE
```bash
=======
| dataset                   | dataset id |
|---------------------------|------------|
| LF-AmazonTitles-131K      | <>         |
| LF-WikiSeeAlsoTitles-131K | <>         |
| LF-AmazonTitles-1.3M      | <>         |

#### RUNNING ECLARE
```
>>>>>>> 49fc87b7eff8ac633cc3a851c34d7dfc3a6c7d6d
cd ${HOME}/scratch/XC/programs/ECLARE
chmod +x run_ECLARE.sh
./run_ECLARE.sh <gpu_id> <ECLARE TYPE> <dataset> <folder name>
e.g.
./run_ECLARE.sh 0 ECLARE LF-AmazonTitles-131K ECLARE_RUN
```