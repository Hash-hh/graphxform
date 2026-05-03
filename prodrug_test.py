def get_prodrug_test_parents():
    """Returns the list of (name, SMILES) tuples used as test parents for the BBB prodrug task."""
    return list(prodrug_parents_test)


def get_prodrug_test_smiles():
    """Returns just the SMILES strings, in the same order as get_prodrug_test_parents()."""
    return [smi for _, smi in prodrug_parents_test]


prodrug_parents_test = [
    # --- Catechols and aminoacid neuro-precursors (canonical lipidization targets) ---
    ("Dopamine",        "C1=CC(=C(C=C1CCN)O)O"),
    ("Norepinephrine",  "C1=CC(=C(C=C1C(CN)O)O)O"),
    ("Epinephrine",     "CNC[C@@H](C1=CC(=C(C=C1)O)O)O"),
    ("L-DOPA",          "C1=CC(=C(C=C1C[C@@H](C(=O)O)N)O)O"),
    ("Carbidopa",       "C[C@@](Cc1ccc(O)c(O)c1)(NN)C(=O)O"),
    ("5-HTP",           "C1=CC2=C(C=C1O)C(=CN2)C[C@@H](C(=O)O)N"),
    ("Tyrosine",        "N[C@@H](Cc1ccc(O)cc1)C(=O)O"),
    ("Tryptophan",      "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O"),
    ("Phenylalanine",   "N[C@@H](Cc1ccccc1)C(=O)O"),
    ("Histidine",       "N[C@@H](Cc1c[nH]cn1)C(=O)O"),

    # --- GABAergic / amino-acid neuromodulators ---
    ("Glycine",         "NCC(=O)O"),
    ("Taurine",         "NCCS(=O)(=O)O"),
    ("Beta-alanine",    "NCCC(=O)O"),
    ("Vigabatrin",      "C=CC(N)CCC(=O)O"),
    ("Pregabalin",      "CC(C)C[C@@H](CN)CC(=O)O"),
    ("Gabapentin",      "NCC1(CC(=O)O)CCCCC1"),
    ("Baclofen",        "NC[C@@H](CC(=O)O)c1ccc(Cl)cc1"),
    ("Tranexamic acid", "NCC1CCC(C(=O)O)CC1"),

    # --- Acidic NSAIDs / analgesics with carboxyl handles ---
    ("Ibuprofen",       "CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O"),
    ("Naproxen",        "COc1ccc2cc([C@@H](C)C(=O)O)ccc2c1"),
    ("Ketoprofen",      "OC(=O)C(C)c1cccc(c1)C(=O)c1ccccc1"),
    ("Diclofenac",      "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"),
    ("Indomethacin",    "Cc1c(CC(=O)O)c2cc(OC)ccc2n1C(=O)c1ccc(Cl)cc1"),
    ("Salicylic acid",  "OC(=O)c1ccccc1O"),
    ("Mefenamic acid",  "Cc1cccc(C)c1Nc1ccccc1C(=O)O"),

    # --- Diuretics / cardiovascular with poor brain entry ---
    ("Bumetanide",      "CCCCNc1cc(C(=O)O)c(S(N)(=O)=O)cc1Oc1ccccc1"),
    ("Furosemide",      "OC(=O)c1cc(NCc2ccco2)c(S(N)(=O)=O)cc1Cl"),
    ("Probenecid",      "CCCN(CCC)S(=O)(=O)c1ccc(C(=O)O)cc1"),
    ("Methyldopa",      "C[C@@](N)(Cc1ccc(O)c(O)c1)C(=O)O"),

    # --- Antibiotics / antifungals / antivirals with CNS gaps ---
    ("Penicillin G",    "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O"),
    ("Ampicillin",      "CC1(C)S[C@@H]2[C@H](NC(=O)[C@@H](N)c3ccccc3)C(=O)N2[C@H]1C(=O)O"),
    ("Cefalexin",       "C[C@H]1[C@H]2SCC(=C(N2C1=O)C(=O)O)C[C@@H](N)c1ccccc1"),
    ("Ciprofloxacin",   "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"),
    ("Levofloxacin",    "C[C@H]1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23"),
    ("Acyclovir",       "Nc1nc2c(ncn2COCCO)c(=O)[nH]1"),
    ("Zidovudine (AZT)","Cc1cn([C@H]2C[C@H](N=[N+]=[N-])[C@@H](CO)O2)c(=O)[nH]c1=O"),
    ("Lamivudine",      "Nc1ccn([C@@H]2CS[C@H](CO)O2)c(=O)n1"),

    # --- Anticancer with documented BBB issues ---
    ("Methotrexate",    "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1"),
    ("Doxorubicin",     "C[C@H]1O[C@H](C[C@H](N)[C@@H]1O)O[C@H]1C[C@@](O)(C(=O)CO)Cc2c(O)c3c(=O)c4cccc(OC)c4c(=O)c3c(O)c12"),
    ("Crizotinib",      "C[C@H](Oc1cc(-c2cnn(C3CCNCC3)c2)cnc1N)c1c(Cl)ccc(F)c1Cl"),
    ("Imatinib",        "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"),

    # --- Endogenous / metabolic agents (high polarity, plausible CNS targets) ---
    ("Glutamic acid",   "N[C@@H](CCC(=O)O)C(=O)O"),
    ("Aspartic acid",   "N[C@@H](CC(=O)O)C(=O)O"),
    ("Cysteine",        "N[C@@H](CS)C(=O)O"),
    ("N-acetylcysteine","CC(=O)N[C@@H](CS)C(=O)O"),
    ("Glutathione",     "N[C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(=O)O)C(=O)O"),
    ("Ascorbic acid",   "OC[C@H](O)[C@H]1OC(=O)C(O)=C1O"),
    ("Niacin",          "OC(=O)c1cccnc1"),
    ("Pyridoxine",      "Cc1ncc(CO)c(CO)c1O"),

    # --- Misc with handles and known/plausible BBB constraints ---
    ("Mesalamine",      "Nc1ccc(O)c(C(=O)O)c1"),
    ("Captopril",       "CC(CS)C(=O)N1CCC[C@H]1C(=O)O"),
    ("Enalaprilat",     "CCOC(=O)[C@H](CCc1ccccc1)N[C@@H](C)C(=O)N1CCC[C@H]1C(=O)O"),
]