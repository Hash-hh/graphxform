from tdc import Oracle


class GSK3BObjective:
    """JNK3 inhibition objective using TDC Oracle."""

    def __init__(self):
        self.oracle = Oracle(name='GSK3B')

    def score(self, smiles: str) -> float:
        """
        Score a single SMILES string for GSK3B inhibition.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            float: JNK3 inhibition score (0 to 1, higher is better)
        """
        try:
            return self.oracle(smiles)
        except:
            return 0.0

    def score_list(self, smiles_list: list) -> list:
        """Score a list of SMILES strings."""
        return [self.score(s) for s in smiles_list]
