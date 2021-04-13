# ECLARE
ECLARE: Extreme Classification with Label Graph Correlations

#### SETUP WORKSPACE
```
mkdir -p ${HOME}/scratch/XC/data 
mkdir -p ${HOME}/scratch/XC/programs
```

#### SETUP ECLARE
```
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
```
cd ${HOME}/scratch/XC/data
gdown --id <dataset id>
unzip *.zip
```
| dataset                   | dataset id |
|---------------------------|------------|
| LF-AmazonTitles-131K      | <>         |
| LF-WikiSeeAlsoTitles-131K | <>         |
| LF-AmazonTitles-1.3M      | <>         |

#### RUNNING ECLARE
```
cd ${HOME}/scratch/XC/programs/ECLARE
chmod +x run_ECLARE.sh
./run_ECLARE.sh <gpu_id> <ECLARE TYPE> <dataset> <folder name>
e.g.
./run_ECLARE.sh 0 ECLARE LF-AmazonTitles-131K ECLARE_RUN
```