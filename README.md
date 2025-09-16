# E3MMDTA: Predicting Drug Target Affinity by Combining E3FP Fingerprints and Multimodal Fusion
Official implementation of "E3MMDTA: Predicting Drug Target Affinity by Combining E3FP Fingerprints and Multimodal Fusion", a novel multimodal deep learning framework for drug-target affinity prediction.
# Overview
![Frame of E3MMDTA](./frame.png)
E3MMDTA is a comprehensive deep learning framework that integrates 3D stereochemical molecular fingerprints (E3FP) with molecular graphs, sequence embeddings, and pre-trained language model features for accurate drug-target affinity (DTA) prediction. Our model addresses limitations of existing methods by effectively capturing 3D molecular information and employing an attention-based fusion module to balance heterogeneous inputs.
# Feature
-3D Molecular Representation: Incorporates E3FP fingerprints to capture stereochemical information beyond conventional 2D descriptors
-Multimodal Fusion: Combines molecular graphs, protein contact graphs, sequence embeddings, and fingerprint vectors
-Advanced Architecture: Utilizes GCN-GAT-SuperGAT pipeline for graph encoding and multi-scale CNN for sequence processing
# Prerequisites
-Python 3.8+
-PyTorch 1.12+
-pyg
-RDKit
-CUDA-capable GPU (recommended)
# install from source
```
#Clone the repository
git clone https://github.com/yourusername/E3MMDTA.git
cd E3MMDTA

# Create a virtual environment (optional but recommended)
conda create -n e3mmdta python=3.8
conda activate e3mmdta

#Install dependencies
pip install -r requirements.txt
```
