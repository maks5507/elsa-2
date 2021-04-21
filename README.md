<p align="center">
    <br>
    <img src="http://maksimeremeev.com/files/elsa.png" width=350/>
    <br>
<p>


# ELSA-2: Extractive Linking of Summarization Approaches

Authors: Maksim Eremeev (mae9785@nyu.edu), Mars Wei-Lun Huang (wh2103@nyu.edu), Phoebe Chen (hc2896@nyu.edu), Anurag Illendula (ai2153@nyu.edu)

This repo introduces an implementation of the ELSA hybrid summarization model. By using extractive attention masking mechanism, we combine large pre-trained transformer-based extractive and abstractive models, such as BART, PEGASUS, and BERTSum.

## Datasets used for experiments

CNN-DailyMail: [Link](https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz), original source: [Link](https://github.com/abisee/cnn-dailymail)

XSum: [Link](http://bollin.inf.ed.ac.uk/public/direct/XSUM-EMNLP18-Summary-Data-Original.tar.gz), original source: [Link](https://github.com/EdinburghNLP/XSum)

#### Downloading & Extracting datasets

```bash
wget https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz
wget http://bollin.inf.ed.ac.uk/public/direct/XSUM-EMNLP18-Summary-Data-Original.tar.gz
wget https://www.dropbox.com/s/cmpfvzxdknkeal4/gazeta_jsonl.tar.gz

tar -xzf cnndm.tar.gz
tar -xzf XSUM-EMNLP18-Summary-Data-Original.tar.gz
tar -xzf gazeta_jsonl.tar.gz
```

## Codestyle check

Before making a commmit / pull-request, please check the coding style by running the bash script in the `codestyle` directory. Make sure that your folder is included in `codestyle/pycodestyle_files.txt` list.

Your changes will not be approved if the script indicates any incongruities (this does not apply to 3rd-party code). 

Usage:

```bash
cd codestyle
sh check_code_style.sh
```

