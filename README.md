# Project Overview
This repository contains the codes required to rerun the experiment "Tackling OOV: Masked Language Models vs Character Diffusion".

## Abstract
LLMs have demonstrated remarkable success; their persistent limitation remains in their ability to generalize effectively in the presence of noisy or out-of-vocabulary (OOV) domains. In this project, I explore the Character Diffusion Model (DLM) as an alternative due to its iterative corrupt-and-reconstruct-sequences strategy in a controlled experiment on the OLID task. Recall and precision indicate that DLM demonstrates a better understanding across both labels, along with the highest F1-score for OOV-prevalent _Offensive_ classification.

Detailed report to be available soon.

## Respective References
- nanoGPT: [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- charDiffisusion: [https://colab.research.google.com/The_Annotated_Discrete_Diffusion_Models.ipynb](https://colab.research.google.com/github/ash80/diffusion-gpt/blob/master/The_Annotated_Discrete_Diffusion_Models.ipynb)
- Base Model Training Dataset: [OmniGEC](https://huggingface.co/datasets/lang-uk/Reddit-MultiGEC/blob/main/reddit_multi_gec.csv)
- Classification task: [SemEval-2019 Task 6: SubTask A](https://aclanthology.org/N19-1144.pdf)

## Working Directory Overview

```
data/
 └── process.py            # codes to create two experimented test set version

dataset/
 └── test/                  # contains two version in .tsv
      ├── base.tsv              # raw text 
      ├── processed.csv         # spelling fixed set
 └── train.tsv                  # raw versions
 └── val.tsv               

discrete_diffusion/
  ├── best_model.pth                # char diffusion base trained model 
  ├── discrete_diffusion.ipynb         # code to obtain the base trained model

logs/            
 └── fine-tuning/                    # models evaluation logs on the classification task
     └── discrete/                     # log on two test set version
        ├── base.txt        
        ├── processed.txt   
     └── nanoGPT/          
        ├── base.txt       
        ├── processed.txt   
 └── foundation training/            # models training logs during base training    
      └── corpus stats.txt       
      └── discrete.txt
      └── nano.log

mis/                                  # Base Training Related Files 
 └── corpus_process.py                  # code to convert input.txt to related training files
 └── input.txt                          # actual dump  
 └── meta.pkl     
 └── train.bin    
 └── val.bin      

nanoGPT_CoLi/                          # nanoGPT base trained model      
 └── out-reddit-clean/                    # models evaluation logs on the classification task
     └── ckpt.pt                           # base trained module

outputs/                               # descriptive results of two models on two test setting
   └── dlm/              
      ├── base_result.tsv        
      ├── processed_result.tsv 
   └── mlm/          
      ├── base_result.tsv  
      ├── processed_result.tsv

src/                                   # experiment relavent codes
   └── module/              
      ├── discrete.py        
      ├── nano.py 
   └── dataset_loader.py
   └── test.py
   └── train.py 

task_pipeline.py                      # experiment root file     
```

## Experimental Step
__Steps__:
1) place your corpus on misc/ naming "input.txt"
	- they will generate the following files: input, train, val, meta.pkl
2) place the output files in the nanoGPT /data/ folder (check their git on the methods to train this models)
3) place the trained model on nanoGPT_coli/ folder
4) now train the discrete character diffusion
	- you can run the .ipynb file inside /discrete_diffusion/ (please refer to the original notebook for parameter understanding)
5) (optional) run the process.py
6) run task_pipeline.py
   - e.g: python task_pipeline.py --epoch 50 --model_type dlm --test_set_type processed
   - kind look into its argument for more controlled experiment

## Citation
```To Be Added```
