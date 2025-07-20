"""
Filters raw ChEMBL SMILES by allowed character set and does a train/validation split.
"""

import time
import random
from tqdm import tqdm

# Prepare and split data into train and val
prepare_data = True
num_validation = 100000

if prepare_data:
    all_smiles = []
    with open(f"./data/chembl/chembl_35_chemreps.txt") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            s = line.rstrip().split("\t")
            smiles = s[1]
            if len(smiles) > 0:
                all_smiles.append(smiles)
        print("Loaded", len(all_smiles), "SMILES")
        print("Shuffling data...")
        random.shuffle(all_smiles)
        print("Saving validation set")
        with open(f"./data/chembl/chembl_valid.smiles", 'w') as f:
            for line in all_smiles[:num_validation]:
                f.write(f"{line}\n")
        print("Saving training set")
        with open(f"./data/chembl/chembl_train.smiles", 'w') as f:
            for line in all_smiles[num_validation:]:
                f.write(f"{line}\n")

print("========")
for datatype in ["train", "valid"]:
    print("Processing", datatype)

    with open(f"./data/chembl/chembl_{datatype}.smiles") as f:
        unfiltered_smiles = [line.rstrip() for line in f]

    allowed_vocabulary = [  # put multi-character occurences first
        "[NH3+]","[SH+]","[C@]","[O+]","[NH+]","[nH+]","[C@@H]","[CH2-]","[C@H]","[NH2+]","[S+]","[CH-]","[S@]","[N-]",
        "[s+]","[nH]","[S@@]","[n+]","[o+]","[NH-]","[C@@]","[S-]","[N+]","[OH+]","[O-]","[n-]",
        "o", "8", "N", "1", "4", "6", "-", ")", "5", "c", "(", "#", "n", "3", "=", "2", "7",
        "C", "O", "S", "s", "F", "P", "p", "Cl", "Br", "I"
    ]

    print("unfiltered:", len(unfiltered_smiles))
    filtered_smiles = []
    for mol in tqdm(unfiltered_smiles):
        temp = mol
        for voc in allowed_vocabulary:
            temp = temp.replace(voc, "")
        if len(temp) == 0:
            filtered_smiles.append(mol)

    print("filtered:", len(filtered_smiles))

    with open(f"./data/chembl/chembl_{datatype}_filtered.smiles", 'w') as f:
        for line in filtered_smiles:
            f.write(f"{line}\n")
