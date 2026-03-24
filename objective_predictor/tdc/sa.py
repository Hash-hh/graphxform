from tdc import Oracle


class SAObjective:
    def __init__(self):
        self.sa = Oracle(name='SA')

    def score(self, smiles):
        """
        Returns the scalar reward for Reinforcement Learning (Sum of properties).
        Range: [0.0, 1.0]
        """
        if not smiles:
            return 0.0


        # SA is [1, 10] where 1 is best. We normalize to [0, 1] where 1 is best.
        raw_sa = self.sa(smiles)
        sa_norm = (10 - raw_sa) / 9.0

        return sa_norm

    def is_successful(self, smiles):
        """
        Strict Binary Evaluation for the 'Success Rate' metric.
        Matches the definition in RationaleRL (Jin et al., 2020).
        """
        # try:
        sa = self.sa(smiles)

        # The hard thresholds defined in the benchmark paper
        return sa < 4.0
        # except:
        #     return False

    def individual_scores(self, smiles):
        """
        Returns individual component scores for analysis.
        """
        sa = self.sa(smiles)
        return {
            "SA": sa
        }

if __name__ == "__main__":
    objective = SAObjective()
    test_smiles = "COc1ccc(-c2ccnc(Nc3ccccc3)n2)cn1"
    # test_smiles = "Cc1cc(-c2ncncc2C[N+](C)(C)CC2CCC2)ccc1-c1ccnc(Nc2ccc(N3CCN(C)CC3)cc2)n1"
    score = objective.score(test_smiles)
    success = objective.is_successful(test_smiles)
    print(f"Score: {score}, Successful: {success}")