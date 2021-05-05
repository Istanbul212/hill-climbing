from tqdm import tqdm

class HC:
    """
    Hill Climbing
    """
    def __init__(self, feature, mutate, evaluate, eval_args=[], steepest=True, minimize=True, stochastic_converge_trials=1000, show_progress=False, cache=False):
        self.feature = feature
        self.mutate = mutate
        self.evaluate = evaluate
        self.args = eval_args
        self.cache = cache
        self.score = evaluate(feature, *eval_args)
        self.trials = stochastic_converge_trials
        self.show_progress = show_progress
        if cache:
            self.cache_dictionary = {
                tuple(feature): self.score
            }
        if steepest and minimize and cache:
            self._train = self._train_min_steepest_ascent_cache
        elif steepest and not minimize and cache:
            self._train = self._train_max_steepest_ascent_cache
        elif not steepest and minimize and cache:
            self._train = self._train_min_stochastic_cache
        elif not steepest and not minimize and cache:
            self._train = self._train_max_stochastic_cache
        elif steepest and minimize and not cache:
            self._train = self._train_min_steepest_ascent_no_cache
        elif steepest and not minimize and not cache:
            self._train = self._train_max_steepest_ascent_no_cache
        elif not steepest and minimize and not cache:
            self._train = self._train_min_stochastic_no_cache
        else:
            self._train = self._train_max_stochastic_no_cache
    
    def _train_min_stochastic_no_cache(self):
        for _ in range(self.trials):
            mutation = self.mutate(self.feature)
            mutation_score = self.evaluate(mutation, *self.args)
            if mutation_score <= self.score:
                self.feature = mutation
                self.score = mutation_score
                return
    
    def _train_max_stochastic_no_cache(self):
        for _ in range(self.trials):
            mutation = self.mutate(self.feature)
            mutation_score = self.evaluate(mutation, *self.args)
            if mutation_score >= self.score:
                self.feature = mutation
                self.score = mutation_score
                return
    
    def _train_min_steepest_ascent_no_cache(self):
        mutations = self.mutate(self.feature)
        self.score, self.feature = min(*zip([self.evaluate(m, *self.args) for m in mutations], mutations), (self.score, self.feature))
    
    def _train_max_steepest_ascent_no_cache(self):
        mutations = self.mutate(self.feature)
        self.score, self.feature = max(*zip([self.evaluate(m, *self.args) for m in mutations], mutations), (self.score, self.feature))
    
    def _train_min_stochastic_cache(self):
        for _ in range(self.trials):
            mutation = self.mutate(self.feature)
            t_mutation = tuple(mutation)
            mutation_score = self.cache_dictionary[t_mutation] if t_mutation in self.cache_dictionary else self.cache_dictionary.setdefault(t_mutation, self.evaluate(mutation, *self.args))
            if mutation_score <= self.score:
                self.feature = mutation
                self.score = mutation_score
                return
    
    def _train_max_stochastic_cache(self):
        for _ in range(self.trials):
            mutation = self.mutate(self.feature)
            t_mutation = tuple(mutation)
            mutation_score = self.cache_dictionary[t_mutation] if t_mutation in self.cache_dictionary else self.cache_dictionary.setdefault(t_mutation, self.evaluate(mutation, *self.args))
            if mutation_score >= self.score:
                self.feature = mutation
                self.score = mutation_score
                return
    
    def _train_min_steepest_ascent_cache(self):
        mutations = self.mutate(self.feature)
        self.score, self.feature = min(*zip([self.cache_dictionary[tuple(m)] if tuple(m) in self.cache_dictionary else self.cache_dictionary.setdefault(tuple(m), self.evaluate(m, *self.args)) for m in mutations], mutations), (self.score, self.feature))

    def _train_max_steepest_ascent_cache(self):
        mutations = self.mutate(self.feature)
        self.score, self.feature = max(*zip([self.cache_dictionary[tuple(m)] if tuple(m) in self.cache_dictionary else self.cache_dictionary.setdefault(tuple(m), self.evaluate(m, *self.args)) for m in mutations], mutations), (self.score, self.feature))
    
    def train(self, trials=0):
        if trials:
            if self.show_progress:
                for _ in tqdm(range(trials), desc="Training..."):
                    self._train()
            else:
                for _ in range(trials):
                    self._train()
        else:
            score_before = self.score
            self._train()
            score_now = self.score
            while score_before != score_now:
                score_before = self.score
                self._train()
                score_now = self.score