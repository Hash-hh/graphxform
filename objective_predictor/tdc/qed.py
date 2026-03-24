from tdc import Oracle


class QEDObjective:
    def __init__(self):
        self.qed = Oracle(name='QED')

    def score(self, smiles):
        """
        Returns the scalar reward for Reinforcement Learning (Sum of properties).
        Range: [0.0, 1.0]
        """
        if not smiles:
            return 0.0

        # QED is [0, 1]
        qed_score = self.qed(smiles)

        return qed_score

    def is_successful(self, smiles):
        """
        Strict Binary Evaluation for the 'Success Rate' metric.
        Matches the definition in RationaleRL (Jin et al., 2020).
        """
        # try:
        qed = self.qed(smiles)

        # The hard thresholds defined in the benchmark paper
        return qed >= 0.6
        # except:
        #     return False

    def individual_scores(self, smiles):
        """
        Returns individual component scores for analysis.
        """
        qed = self.qed(smiles)
        return {
            "QED": qed,
        }

if __name__ == "__main__":
    objective = QEDObjective()
    test_smiles = "COc1ccc(-c2ccnc(Nc3ccccc3)n2)cn1"
    # test_smiles = "Cc1cc(-c2ncncc2C[N+](C)(C)CC2CCC2)ccc1-c1ccnc(Nc2ccc(N3CCN(C)CC3)cc2)n1"
    score = objective.score(test_smiles)
    success = objective.is_successful(test_smiles)
    print(f"Score: {score}, Successful: {success}")