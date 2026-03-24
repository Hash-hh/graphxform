from tdc import Oracle


class GSKObjective:
    def __init__(self):
        self.gsk3 = Oracle(name='GSK3B')

    def score(self, smiles):
        """
        Returns the scalar reward for Reinforcement Learning (Sum of properties).
        Range: [0.0, 1.0]
        """
        if not smiles:
            return 0.0

        # Get raw scores
        # GSK3B and JNK3 return probabilities [0, 1]
        gsk_score = self.gsk3(smiles)

        return gsk_score

    def is_successful(self, smiles):
        """
        Strict Binary Evaluation for the 'Success Rate' metric.
        Matches the definition in RationaleRL (Jin et al., 2020).
        """
        # try:
        gsk = self.gsk3(smiles)

        # The hard thresholds defined in the benchmark paper
        return gsk >= 0.5
        # except:
        #     return False

    def individual_scores(self, smiles):
        """
        Returns individual component scores for analysis.
        """
        gsk = self.gsk3(smiles)
        return {
            "GSK3B": gsk,
        }

if __name__ == "__main__":
    objective = GSKObjective()
    test_smiles = "COc1ccc(-c2ccnc(Nc3ccccc3)n2)cn1"
    # test_smiles = "Cc1cc(-c2ncncc2C[N+](C)(C)CC2CCC2)ccc1-c1ccnc(Nc2ccc(N3CCN(C)CC3)cc2)n1"
    score = objective.score(test_smiles)
    success = objective.is_successful(test_smiles)
    print(f"Score: {score}, Successful: {success}")