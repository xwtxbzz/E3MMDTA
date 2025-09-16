from rdkit import Chem
from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer
from skfp.fingerprints import *

# dataset = 'kiba'
def get_fingerprint(smiles):
    mol_from_smiles = MolFromSmilesTransformer()
    mols = mol_from_smiles.transform([smiles])
    fp = ERGFingerprint()
    ergfp = fp.transform(mols)[0]
    fp = PubChemFingerprint()
    pubfp = fp.transform(mols)[0]
    fp = MACCSFingerprint()
    maccsfp = fp.transform(mols)[0]
    try:
        conf_gen = ConformerGenerator()
        mols = conf_gen.transform(mols)
        fp = E3FPFingerprint()
        e3fp = fp.transform(mols)[0]
    except:
        e3fp = np.zeros(1024)
        
    return (e3fp, ergfp, pubfp, maccsfp)

import pandas as  pd
import numpy as np
import os
dataset = 'KIBA'
# smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)Oc1ccccc1C(=O)O"]
drug = pd.read_csv("./data/{}/smiles.csv".format(dataset)).to_numpy()
for idx,s in enumerate(drug):
    print(idx,s[1])
    if len(s[1]) > 500:
        try:
            finger = np.load('./data/{}/{}/{}.npz'.format('ttd/KI','drug_fingerprints', s[0]), allow_pickle=True)
            np.savez('./data/{}/{}/{}.npz'.format(dataset,'drug_fingerprints', s[0]), e3fp = finger['e3fp'],ergfp = finger['ergfp'],pubfp = finger['pubfp'],maccsfp = finger['maccsfp'])
        except:
            continue
    if os.path.exists('./data/{}/{}/{}.npz'.format(dataset,'drug_fingerprints', s[0])):
        continue
    e3fp,ergfp,pubfp,maccsfp = get_fingerprint(s[1])

    np.savez('./data/{}/{}/{}.npz'.format(dataset,'drug_fingerprints', s[0]), e3fp = e3fp,ergfp = ergfp,pubfp = pubfp,maccsfp = maccsfp)