from tqdm import tqdm

class HC:
    """
    Hill Climbing
    """
    def __init__(self, feature, mutate, evaluate, eval_args=[], steepest=True, minimize=True, stochastic_converge_trials=1000, show_progress=False):
        self.feature = feature
        self.mutate = mutate
        self.evaluate = evaluate
        self.args = eval_args
        self.score = evaluate(feature, *eval_args)
        self.trials = stochastic_converge_trials
        self.show_progress = show_progress
        if steepest and minimize:
            self._train = self._train_min_steepest_ascent
        elif steepest and not minimize:
            self._train = self._train_max_steepest_ascent
        elif not steepest and minimize:
            self._train = self._train_min_stochastic
        else:
            self._train = self._train_max_stochastic
    def _train_min_stochastic(self):
        for _ in range(self.trials):
            mutation = self.mutate(self.feature)
            mutation_score = self.evaluate(mutation, *self.args)
            if mutation_score <= self.score:
                self.feature = mutation
                self.score = mutation_score
                return
    def _train_max_stochastic(self):
        for _ in range(self.trials):
            mutation = self.mutate(self.feature)
            mutation_score = self.evaluate(mutation, *self.args)
            if mutation_score >= self.score:
                self.feature = mutation
                self.score = mutation_score
                return
    def _train_min_steepest_ascent(self):
        mutations = self.mutate(self.feature)
        self.score, self.feature = min(*zip([self.evaluate(m, *self.args) for m in mutations], mutations), (self.score, self.feature))
    def _train_max_steepest_ascent(self):
        mutations = self.mutate(self.feature)
        self.score, self.feature = max(*zip([self.evaluate(m, *self.args) for m in mutations], mutations), (self.score, self.feature))
    def train(self, trials=1):
        if self.show_progress:
            for _ in tqdm(range(trials), desc="Training..."):
                self._train()
        else:
            for _ in range(trials):
                self._train()