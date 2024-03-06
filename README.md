# Code for InfoCTM: A Mutual Information Maximization Perspective of Cross-lingual Topic Modeling (AAAI2023)

## **Check our latest topic modeling toolkit [TopMost](https://github.com/bobxwu/topmost) !**

[PDF](https://arxiv.org/pdf/2304.03544.pdf)

## Usage

### 1. Prepare Environment

    python=3.7
    torch==1.7.1
    scikit-learn==1.0.2
    gensim==4.0.1
    pyyaml==6.0
    spacy==2.3.2

### 2. Training

We provide a shell script for training:

    ./run.sh


### 3. Evaluation

**Topic coherence**:

We have released the implementation of [CNPMI](https://github.com/BobXWu/CNPMI).

**Topic diversity**:

We use the average $TU$ score of two langauges:

    python utils/TU.py --path {path of topic words in language 1}
    python utils/TU.py --path {path of topic words in language 2}


## Citation

If you want to use our code, please cite as

    @article{wu2023infoctm,
    title={InfoCTM: A Mutual Information Maximization Perspective of Cross-Lingual Topic Modeling},
    author={Wu, Xiaobao and Dong, Xinshuai and Nguyen, Thong and Liu, Chaoqun and Pan, Liangming and Luu, Anh Tuan},
    journal={arXiv preprint arXiv:2304.03544},
    year={2023}
    }

